"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

FALCON: Lightweight and Accurate Convolution

File: utils/timer.py
 - Contain source code for a timer.
 - Code is got from
 https://dev.tencent.com/u/zzpu/p/yolov2/git/raw/
 4b2c6c7df1876363aba3bbd600aa68e4deeb4487/utils/timer.py

Version: 1.0
"""

import time


class Timer:
    """A simple timer."""
    def __init__(self):
        """
        Initialize timer.
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        """
        Start timer.
        """
        self.start_time = time.time()

    def toc(self, average=True):
        """
        Stop timer.
        
        :param average: whether to calculate the average time or not
        :return: average_time: average time
        """
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time

        return self.diff
