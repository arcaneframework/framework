﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/String.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/StringView.h"

#include <vector>

using namespace Arccore;

TEST(String, Misc)
{
  String e = "Titi";
  String f = "Toto23";
  ASSERT_TRUE(f.endsWith("23")) << "Bad compare 1";
  ASSERT_TRUE(f.startsWith("Toto")) << "Bad compare 2";
  ASSERT_FALSE(f.startsWith("Toto1")) << "Bad compare 3";
  ASSERT_FALSE(f.endsWith("Toto1")) << "Bad compare 4";
  ASSERT_FALSE(f.startsWith("Toto234")) << "Bad compare 5";

  ASSERT_FALSE(f.endsWith("Toto234")) << "Bad compare 6";

  String s2 = f.substring(3);
  ASSERT_TRUE(s2=="o23") << "Bad compare 7";

  s2 = f.substring(3,2);
  std::cout << "S2_8=" << s2 << '\n';
  ASSERT_FALSE(s2!="o2") << "Bad compare 8";

  s2 = f.substring(1,2);
  std::cout << "S2_9=" << s2 << '\n';
  ASSERT_FALSE(s2!="ot") << "Bad compare 9";

  s2 = f.substring(7,2);
  std::cout << "S2_10=" << s2 << '\n';
  ASSERT_FALSE(s2!="") << "Bad compare 10";

  s2 = f.substring(2,1);
  std::cout << "S2_11=" << s2 << '\n';
  ASSERT_FALSE(s2!="t") << "Bad compare 11";

  s2 = f.substring(5,1);
  std::cout << "S2_12=" << s2 << '\n';
  ASSERT_FALSE(s2!="3") << "Bad compare 12";

  s2 = f.substring(0);
  std::cout << "S2_13=" << s2 << '\n';
  ASSERT_FALSE(s2!=f) << "Bad compare 13";

  String g = "   \tceci   \tcela ";
  std::cout << " G=  '" << g << "'" << '\n';
  String g2 = String::collapseWhiteSpace(g);
  std::cout << " G2= '" << g2 << "'" << '\n';
  String g3 = String::replaceWhiteSpace(g);
  std::cout << " G3= '" << g3 << "'" << '\n';
  String expected_g3 ="    ceci    cela ";
  ASSERT_EQ(g3,expected_g3);
  String expected_g2 ="ceci cela";
  ASSERT_EQ(g2,expected_g2);

  String gnull;
  String gnull2 = String::collapseWhiteSpace(gnull);
  std::cout << "GNULL2='" << gnull2 << "'" << '\n';
  ASSERT_EQ(gnull2,String());

  String gempty("");
  String gempty2 = String::collapseWhiteSpace(gempty);
  std::cout << "GEMPTY2='" << gempty2 << "'" << '\n';
  String expected_gempty2 = "";
  ASSERT_EQ(gempty2,expected_gempty2);

  {
    String knull;
    String kempty { gempty };
    String k1 { "titi" };
    String k2 { "ti" };
    String k3 { "to" };
    ASSERT_TRUE(knull.contains(gnull)) << "Bad null contains null";
    ASSERT_FALSE(knull.contains(gempty)) << "Bad null contains empty";
    ASSERT_TRUE(kempty.contains(gnull)) << "Bad empty contains null";
    ASSERT_TRUE(kempty.contains(gempty)) << "Bad empty contains null";
    ASSERT_TRUE(k1.contains(gnull)) << "Bad null contains null";
    ASSERT_TRUE(k1.contains(gempty)) << "Bad contains empty";
    ASSERT_TRUE(k1.contains(k2)) << "Bad k1 contains k2";
    ASSERT_FALSE(k2.contains(k1)) << "Bad k2 contains k1";
    ASSERT_FALSE(k1.contains(k3)) << "Bad k1 contains k3";
  }
  {
    String k0 = ":Toto::Titi:::Tata::::Tutu:Tete:";
    //String k0 = ":Toto:Titi";
    //String k0 = ":Toto::Titi";
    std::cout << "ORIGINAL STRING TO STRING = '" << k0 << "'" << '\n';
    std::vector<String> k0_list;
    k0.split(k0_list,':');
    for( size_t i=0, n=k0_list.size(); i<n; ++i ){
      std::cout << "K i=" << i << " v='" << k0_list[i] << "' is_null?=" << k0_list[i].null() << '\n';
    }
    ASSERT_EQ(k0_list[0],String(":Toto"));
    ASSERT_EQ(k0_list[1],String(":Titi"));
    ASSERT_EQ(k0_list[2],String(":"));
    ASSERT_EQ(k0_list[3],String("Tata"));
    ASSERT_EQ(k0_list[4],String(":"));
    ASSERT_EQ(k0_list[5],String(":Tutu"));
    ASSERT_EQ(k0_list[6],String("Tete"));
  }
}

TEST(String, StdStringView)
{
  const char* ref1 = "S1éà";
  const char* ref2 = "ù*aXZáé";
  // Ref3 = Ref1 + Ref2
  const char* ref3 = "S1éàù*aXZáé";
  std::string std_ref3 { ref3 };
  String snull;
  String sempty { "" };
  String s1 = ref1;
  String s2 = ref2;
  String s3 = ref1;
  s3 = s3 + ref2;
  std::cout << "S1 '" << s1 << "'_SIZE=" << s1.length() << '\n';
  std::cout << "S2 '" << s2 << "'_SIZE=" << s2.length() << '\n';
  std::cout << "S3 '" << s3 << "'_SIZE=" << s3.length() << '\n';
  std::string_view vempty = sempty.toStdStringView();
  ASSERT_EQ((Int64)vempty.size(),0) << "vempty.size()==0";
  std::string_view vnull = snull.toStdStringView();
  ASSERT_EQ((Int64)vnull.size(),0) << "vnull.size()==0";
  std::string_view v1 = s1.toStdStringView();
  ASSERT_EQ(v1,ref1) << "v1==ref1";
  std::string_view v2 = s2.toStdStringView();
  ASSERT_EQ(v2,ref2) << "v2==ref2";
  std::string_view v3 = s3.toStdStringView();
  ASSERT_EQ(v3,std_ref3) << "v3==ref3";

  String s4 = s3 + snull;
  std::string_view v4 = s4.toStdStringView();
  ASSERT_EQ(v4,v3) << "v4==v3";

  String s5 = s3 + sempty;
  std::string_view v5 = s5.toStdStringView();
  ASSERT_EQ(v5,v4) << "v5==v4";

  String s6 = s2;
  const char* t1 = "testà1";
  const char* t2 = "testé2";
  std::string st1 = t1;
  std::string_view st1v = st1;
  std::string st1_2 = st1 + t2;
  s6 = t1;
  ASSERT_EQ(s6.toStdStringView(),st1v) << "s6==st1";

  String s7 = s3;
  s7 = st1_2;
  ASSERT_EQ(s7,st1_2) << "s7==st1_2";
  String s8 = s6 + t2;
  ASSERT_EQ(s8,st1_2) << "s8==st1_2";
  String s9 = s7 + snull;
  ASSERT_EQ(s9,st1_2) << "s9==st1_2";
  String s10 = s7 + sempty;
  ASSERT_EQ(s10,st1_2) << "s10==st1_2";
}
