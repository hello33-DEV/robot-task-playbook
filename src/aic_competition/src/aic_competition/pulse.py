"""Threaded pulse worker. All STOP paths are independent of mission ROS callbacks."""
import threading
import time
import math
import uuid


class PulseDriver:
    def __init__(self, config, transport=None, clock=time.monotonic):
        self.c, self.transport, self.clock = config, transport, clock
        self.start_frame = bytes.fromhex(config['start_hex'])
        self.stop_frame = bytes.fromhex(config['stop_hex'])
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.armed = False
        self.busy = False
        self.failed = None
        self.next_allowed = float('-inf')
        self.thread = None
        self.pulse_id = None
        self.pulse_result = 'idle'

    def snapshot(self):
        with self.lock:
            return dict(armed=self.armed, busy=self.busy, fault=self.failed,
                        pulse_id=self.pulse_id, pulse_result=self.pulse_result)

    def _write(self, frame):
        if self.transport is not None:
            count = self.transport.write(frame)
            if count != len(frame):
                raise IOError('short serial write %s/%s' % (count, len(frame)))

    def arm(self, enabled):
        with self.lock:
            if enabled and self.failed:
                raise RuntimeError('serial driver fault is latched: ' + self.failed)
            if enabled and self.busy:
                raise RuntimeError('cannot arm during pulse')
            self.armed = bool(enabled)
            if not enabled:
                self.cancel_event.set()
                try:
                    self._write(self.stop_frame)
                except Exception as exc:
                    self.failed = str(exc)
                    self.pulse_result = 'failed'
                    raise

    def request(self, interlock, pulse_seconds=None):
        duration = self.c['pulse_seconds'] if pulse_seconds is None else pulse_seconds
        if not math.isfinite(duration) or not 0 < duration <= .15:
            return False
        with self.lock:
            if not self.armed or self.failed or self.busy or self.clock() < self.next_allowed or not interlock():
                return False
            self.busy = True
            self.pulse_id = uuid.uuid4().hex
            self.pulse_result = 'pending'
            self.cancel_event.clear()
            self.thread = threading.Thread(target=self._pulse, args=(interlock, duration), daemon=True)
            try:
                self.thread.start()
            except Exception as exc:
                self.failed, self.armed, self.busy = str(exc), False, False
                self.pulse_result = 'failed'
                self.thread = None
                return False
            return True

    def _pulse(self, interlock, duration):
        completed = False
        try:
            with self.lock:
                if not self.armed or self.cancel_event.is_set() or not interlock():
                    return
                self._write(self.start_frame)
            end = self.clock() + duration
            while self.clock() < end:
                if self.cancel_event.wait(min(.01, max(0, end-self.clock()))) or not interlock():
                    break
            else:
                completed = interlock()
        except Exception as e:
            with self.lock:
                self.failed, self.armed = str(e), False
        finally:
            with self.lock:
                try:
                    self._write(self.stop_frame)
                except Exception as e:
                    self.failed, self.armed = str(e), False
                completed = completed and self.armed and not self.cancel_event.is_set()
                self.pulse_result = 'failed' if self.failed else 'completed' if completed else 'cancelled'
                if self.pulse_result != 'completed':
                    self.armed = False
                self.next_allowed = self.clock() + self.c['cooldown_seconds']
                self.busy = False

    def stop(self):
        self.cancel_event.set()
        try:
            self.arm(False)
        except Exception as e:
            with self.lock:
                self.failed, self.armed = str(e), False

    def close(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=self.c['write_timeout'] * 3 + .5)
        with self.lock:
            if self.transport is not None:
                self.transport.close()
