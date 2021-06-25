/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>

#include <arccore/base/Exception.h>
#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/trace/ITraceMng.h>

#include <alien/AlienTestExport.h>

namespace AlienTest
{

class ALIEN_TEST_FRAMEWORK_EXPORT Environment
{
 public:
  static void initialize(int argc, char** argv);

  static void finalize();

  static Arccore::MessagePassing::IMessagePassingMng* parallelMng();

  static Arccore::ITraceMng* traceMng();

  template <typename T>
  static int execute(int argc, char** argv, T&& t)
  {
    initialize(argc, argv);

    int ret = 0;
    try {
      ret = t();
    }
    catch (const Arccore::Exception& ex) {
      std::cerr << "Exception: " << ex << '\n';
      ret = 3;
    }
    catch (const std::exception& ex) {
      std::cerr << "** A standard exception occurred: " << ex.what() << ".\n";
      ret = 2;
    }
    catch (...) {
      std::cerr << "** An unknown exception has occured...\n";
      ret = 1;
    }

    finalize();

    return ret;
  }
};
} // namespace AlienTest
