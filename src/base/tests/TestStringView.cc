// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/StringView.h"
#include "arccore/base/String.h"

#include <iostream>

using namespace Arccore;

TEST(StringView, StdStringView)
{
  const char* ref1 = "S1éà";
  const char* ref2 = "ù*aXZáé";
  // Ref3 = Ref1 + Ref2
  const char* ref3 = "S1éàù*aXZáé";
  std::string std_ref3 { ref3 };
  String snull;
  String sempty { "" };
  StringView s1 = ref1;
  StringView s2 = ref2;
  String s3 = ref1;
  s3 = s3 + ref2;
  std::cout << "S2 '" << s2 << "'_SIZE=" << s2.length() << '\n';
  std::cout << "S3 '" << s3 << "'_SIZE=" << s3.length() << '\n';
  std::string_view vempty = sempty.toStdStringView();
  ASSERT_EQ((Int64)vempty.size(),0) << "vempty.size()==0";
  std::string_view vnull = snull.toStdStringView();
  ASSERT_EQ((Int64)vnull.size(),0) << "vnull.size()==0";
  std::string_view v1 = s1.toStdStringView();
  std::cout << "S1 '" << s1 << "'_SIZE=" << s1.length() << " V1 " << v1.size() << " ='" << v1 << "'" << '\n';
  ASSERT_EQ(v1,ref1) << "v1==ref1";
  std::string_view v2 = s2.toStdStringView();
  std::cout << "S2 '" << s2 << "'_SIZE=" << s2.length() << " V2 " << v2.size() << " ='" << v2 << "'" << '\n';
  ASSERT_EQ(v2,ref2) << "v2==ref2";
  std::string_view v3 = s3.toStdStringView();
  std::cout << "S3 '" << s3 << "'_SIZE=" << s3.length() << " V3 " << v3.size() << " ='" << v3 << "'" << '\n';
  ASSERT_EQ(v3,std_ref3) << "v3==ref3";

  String s4 = s3 + snull;
  std::cout << "S4 '" << s4 << "'_SIZE=" << s4.length() << '\n';
  ASSERT_EQ(s4.length(),s3.length());
  StringView v4 = s4.view();
  std::cout << "S4 '" << s4 << "'_SIZE=" << s4.length() << " V4 " << v4.size() << " ='" << v4 << "'" << '\n';
  ASSERT_EQ(v4,v3) << "v4==v3";

  String s5 = s3 + sempty;
  std::cout << "S5 '" << s5 << "'_SIZE=" << s5.length() << '\n';
  StringView v5 = s5.view();
  std::cout << "S5 '" << s5 << "'_SIZE=" << s5.length() << " V5 " << v5.size() << " ='" << v5 << "'" << '\n';
  ASSERT_EQ(v5,v4) << "v5==v4";
}
