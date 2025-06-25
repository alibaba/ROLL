# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import time
from collections import defaultdict


class ThreadSafeDict:
    def __init__(self):
        self._data = {}
        self._conditions = defaultdict(threading.Condition)
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            condition = self._conditions[key]
            with condition:
                condition.notify_all()

    def get(self, key, timeout=None):
        with self._lock:
            if key in self._data:
                return self._data[key]
            condition = self._conditions[key]

        with condition:
            waited = condition.wait(timeout=timeout)
            with self._lock:
                if key in self._data:
                    return self._data[key]
                else:
                    raise KeyError(f"Key '{key}' was not set within the timeout period.")

    def pop(self, key, timeout=None):
        ret = self.get(key, timeout=timeout)
        self.remove(key)
        return ret

    def contains(self, key):
        with self._lock:
            return key in self._data

    def remove(self, key):
        with self._lock:
            if key in self._data:
                del self._data[key]
            if key in self._conditions:
                del self._conditions[key]

    def clear(self):
        with self._lock:
            self._data.clear()
            self._conditions.clear()

    def keys(self):
        with self._lock:
            return self._data.keys()

    def __len__(self):
        with self._lock:
            return len(self._data)

    def __contains__(self, key):
        return self.contains(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __delitem__(self, key):
        self.remove(key)