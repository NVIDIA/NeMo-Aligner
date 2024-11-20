#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/test_cases

set -u

# Define ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

for script in $(ls | grep -v '\.log$'); do
  echo -n "[Running] $script..."
  
  start_time=$(date +%s.%N)
  output=$(bash "$script" 2>&1)
  exit_code=$?
  end_time=$(date +%s.%N)
  elapsed=$(echo "$end_time $start_time" | awk '{print $1 - $2}')

  if [[ $exit_code -eq 0 ]]; then
    echo -e "${GREEN}PASSED${NC} (Time: ${elapsed}s)"
  else
    echo -e "${RED}FAILED${NC} (Time: ${elapsed}s)"
    echo -e "${YELLOW}"
    echo "$output" | tail -n 10
    echo -e "${NC}"
  fi
done
