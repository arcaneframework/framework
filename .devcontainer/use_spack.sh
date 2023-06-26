#! /usr/bin/env bash
#
# Copyright 2021 IFPEN-CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#  SPDX-License-Identifier: Apache-2.0
#
ENV_FILE=/tmp/foo

[[ $# -ge 1 ]] && ENV_FILE="$1"

. /spack/share/spack/setup-env.sh && spack env activate -V alien && spack load --only dependencies --sh alien >"${ENV_FILE}"

# shellcheck disable=SC1090
. "$ENV_FILE"

# googletest is a build only dependency
spack load googletest
