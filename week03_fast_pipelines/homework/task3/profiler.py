import json
import time
import torch
import os
from collections import defaultdict
import contextlib

class Profile:
    def __init__(self, model, name="model", schedule=None, sync_cuda=False):
        self.model = model
        self.name_map = self._build_name_map(model, name)
        self.events = []             # список собранных событий (dict)
        self._handles = []           # зарегистрированные hook handles
        self._fwd_start = {}         # модуль -> ns (начало forward)
        self._bwd_start = {}         # модуль -> ns (начало backward)
        self._step_idx = 0
        self.schedule = schedule
       
        self.sync_cuda = sync_cuda
        if sync_cuda is None:
            sync_cuda = torch.cuda.is_available()
        self._t0_ns = None
      
    
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return not list(module.children())
    
    def _now_ns(self):
        # при необходимости синхронизируемся с CUDA, чтобы учесть завершение всех кёрнелов
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter_ns()

    def _forward_pre_hook(self, module, inputs):
        # фиксируем старт forward для модуля
        self._fwd_start[module] = self._now_ns()

    def _forward_post_hook(self, module, inputs, outputs):
        # фиксируем окончание forward и пишем событие
        t_end = self._now_ns()
        t_start = self._fwd_start.pop(module, None)
        if t_start is None:
            return
        self._emit_event(module, phase="forward", t_start_ns=t_start, t_end_ns=t_end)

    def _backward_pre_hook(self, module, grad_output):
        # старт backward для модуля
        self._bwd_start[module] = self._now_ns()

    def _backward_post_hook(self, module, grad_input, grad_output):
        # конец backward
        t_end = self._now_ns()
        t_start = self._bwd_start.pop(module, None)
        if t_start is None:
            return
        self._emit_event(module, phase="backward", t_start_ns=t_start, t_end_ns=t_end)

    def _emit_event(self, module, phase: str, t_start_ns: int, t_end_ns: int):
        # нормализуем таймстемпы к началу профайла и переводим в микросекунды
        base = self._t0_ns if self._t0_ns is not None else t_start_ns
        ts_us = (t_start_ns - base) / 1000.0
        dur_us = (t_end_ns - t_start_ns) / 1000.0
        name = self.name_map.get(module, module.__class__.__name__)
        self.events.append({
            "name": name,
            "cat": phase,          # forward/backward
            "ts_us": ts_us,
            "dur_us": dur_us,
            "step": self._step_idx,
        })

    def __enter__(self):
        # запоминаем базовое время, чтобы таймстемпы были от нуля
        self._t0_ns = time.perf_counter_ns()
        # регистрируем хуки на все модули
        for m in self.name_map.keys():
            self._handles.append(m.register_forward_pre_hook(self._forward_pre_hook))
            self._handles.append(m.register_forward_hook(self._forward_post_hook))
            # full backward hooks работают на AutogradNode уровня модуля
            self._handles.append(m.register_full_backward_pre_hook(self._backward_pre_hook))
            self._handles.append(m.register_full_backward_hook(self._backward_post_hook))
        return self

    def __exit__(self, type, value, traceback):
        # снять хуки
        for h in self._handles:
            with contextlib.suppress(Exception):
                h.remove()
        self._handles.clear()

    def step(self):
        """
        Маркер итерации (необязательно).
        Удобно логически отделять итерации/фазы и потом фильтровать события по step.
        """
        self._step_idx += 1

    def summary(self):
        print("Summary (first 50 events):")
        for e in self.events[:50]:
            print(f"[{e['cat']}] {e['name']}  dur={e['dur_us']:.3f}us")


    def to_perfetto(self, path="trace.json"):
        """
        Сохраняет события в формат Chrome Tracing / Perfetto.
        Открой через chrome://tracing или ui.perfetto.dev
        """
        trace = {
            "traceEvents": []
        }
        pid = os.getpid()
        # соберём по фазам разные threads (чтобы было легче визуально отличать)
        tid_map = {"forward": 0, "backward": 1, "meta": 2}

        for ev in self.events:
            item = {
                "name": ev["name"],
                "cat": ev["cat"],                # forward/backward/meta
                "ph": "X",                       # complete event
                "ts": ev["ts_us"],               # microseconds
                "dur": ev["dur_us"],             # microseconds
                "pid": pid,
                "tid": tid_map.get(ev["cat"], 0),
                "args": {"step": ev.get("step", 0)}
            }
            trace["traceEvents"].append(item)

        with open(path, "w") as f:
            json.dump(trace, f)
        print(f"Perfetto trace saved to: {path}")

