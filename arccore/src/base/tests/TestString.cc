// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/String.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/StringView.h"
#include "arccore/base/StringUtils.h"
#include "arccore/base/CoreArray.h"
#include "arccore/base/BasicTranscoder.h"

#include <vector>
#include <array>

#ifdef ARCCORE_OS_WIN32
#include <windows.h>
#endif

using namespace Arcane;
using namespace Arcane::Impl;

namespace
{
// Classe pour restaurer automatiquement les flags() d'un 'std::ostream'
  // To Test
class IosFlagsWrapper
{
 public:

  explicit IosFlagsWrapper(std::ostream* o)
  : m_stream(o)
  , m_flags(o->flags())
  {}
  ~IosFlagsWrapper() { m_stream->flags(m_flags); }

 private:

  std::ostream* m_stream;
  std::ios_base::fmtflags m_flags;
};

#ifdef ARCCORE_OS_WIN32
void _testWStringWin32(const char* str0)
{
  std::wstring wideWhat;
  int str0len = (int)std::strlen(str0);
  int convertResult = MultiByteToWideChar(CP_UTF8, 0, str0, str0len, NULL, 0);
  if (convertResult <= 0) {
    wideWhat = L"Exception occurred: Failure to convert its message text using MultiByteToWideChar: convertResult=";
  }
  else {
    wideWhat.resize(convertResult + 10);
    convertResult = MultiByteToWideChar(CP_UTF8, 0, str0, str0len, wideWhat.data(), (int)wideWhat.size());
    if (convertResult <= 0) {
      wideWhat = L"Exception occurred: Failure to convert its message text using MultiByteToWideChar: convertResult=";
    }
    else {
      wideWhat.resize(convertResult);
      //wideWhat.insert(0, L"Exception occurred: ");
    }
  }
  std::cout << "STR0=" << str0 << " len=" << str0len << "\n";
  std::cout << "convertResult=" << convertResult << "\n";
  std::wcout << "WSTR0 len=" << wideWhat.length() << " v='" << wideWhat << "'\n ";
  for (int i = 0; i < convertResult; ++i)
    std::wcout << "PRINT1 I=" << i << " V=" << wideWhat[i] << "\n";
  std::wcout.flush();
  for (int i = 0; i < convertResult; ++i)
    std::cout << "PRINT2 I=" << i << " V=" << std::hex << static_cast<int>(wideWhat[i]) << "\n";
}
#endif

void _doConvertTest(const char* name, const String& str)
{
  std::cout << "Name=" << name << "\n";
  std::cout << "OrigUtf8 size=" << str.utf8() << "\n";
  {
    CoreArray<Byte> utf8_orig_bytes(str.bytes());
    CoreArray<Byte> utf8_final_bytes;
    CoreArray<UChar> utf16_bytes;
    BasicTranscoder::transcodeFromUtf8ToUtf16(utf8_orig_bytes, utf16_bytes);
    BasicTranscoder::transcodeFromUtf16ToUtf8(utf16_bytes, utf8_final_bytes);
    std::cout << "OrigBytes=" << utf8_orig_bytes.constView() << "\n";
    std::cout << "FinalTranscoderUtf16=" << utf16_bytes.constView() << "\n";
    std::cout << "FinalTranscoderUtf8=" << utf8_final_bytes.constView() << "\n";
    ASSERT_EQ(utf8_orig_bytes.constView(), utf8_final_bytes.constView());
  }
  String str2 = String::collapseWhiteSpace(str);
  ASSERT_EQ(str, str2);

  std::vector<UChar> utf16_vector{ StringUtils::asUtf16BE(str) };
  Span<const UChar> utf16_bytes(utf16_vector.data(), utf16_vector.size());
  std::cout << "Utf16 bytes = " << utf16_bytes << "\n";
  String str3(utf16_bytes.smallView());
  std::cout << "ToUtf8 size=" << str3.bytes() << "\n";
  ASSERT_EQ(str, str3);
}

} // namespace

// TODO: Regarder pourquoi le test ne passe pas sous windows sur le CI de github
// (alors qu'il fonctionne sur un PC perso. Il faudrait regarder si cela n'est pas
// un problème d'encodage par défaut).
TEST(String, Utf8AndUtf16)
{
  IosFlagsWrapper io_wrapper(&std::cout);
  {
    String str1_direct = "▲▼●■◆éà😀a😈";
    std::cout << "STR1_UTF8=" << str1_direct << "\n";

    std::array<Byte, 28> str1_bytes = { 0xE2, 0x96, 0xB2, 0xE2, 0x96, 0xBC, 0xE2, 0x97, 0x8F, 0xE2, 0x96, 0xA0, 0xE2, 0x97,
                                        0x86, 0xC3, 0xA9, 0xC3, 0xA0, 0xF0, 0x9F, 0x98, 0x80, 0x61, 0xF0, 0x9F, 0x98, 0x88 };
    String str1(str1_bytes);
    String str1_orig("▲▼●■◆");
    //String str1(str1_bytes);
#ifdef ARCCORE_OS_WIN32
    _testWStringWin32(str1.localstr());
#endif
    std::vector < UChar> utf16_vector_direct{ StringUtils::asUtf16BE(str1_direct) };
    std::vector<UChar> utf16_vector{ StringUtils::asUtf16BE(str1) };
    std::vector<UChar> big_endian_ref_vector{ 0x25b2, 0x25bc, 0x25cf, 0x25a0, 0x25c6, 0x00E9, 0x00E0, 0xD83D, 0xDE00, 0x0061, 0xD83D, 0xDE08 };
    //std::vector<UChar> big_endian_ref_vector{ 0x25b2, 0x25bc, 0x25cf, 0x25a0, 0x25c6 };
    for (int x : utf16_vector)
      std::cout << "Utf16: " << std::hex << x << "\n";
    ASSERT_EQ(big_endian_ref_vector, utf16_vector_direct);
    ASSERT_EQ(big_endian_ref_vector, utf16_vector);
    Span<const UChar> utf16_bytes(utf16_vector.data(), utf16_vector.size());
    std::cout << "Utf16_size=" << utf16_bytes.smallView() << "\n";

    std::cout << "BEFORE_CREATE_STR2\n";
    String str2(utf16_bytes.smallView());
    std::cout << "str1.utf16=" << str1.utf16() << "\n";
    std::cout << "str2.utf16=" << str2.utf16() << "\n";

    bool is_same = (str1 == str2);

    std::cout << "is_same=" << is_same << "\n";
    ASSERT_EQ(str1, str2);

    std::cout << "str1.utf16=" << str1.utf16() << "\n";
    std::cout << "str2.utf16=" << str2.utf16() << "\n";
    std::cout.flush();

    ASSERT_EQ(str1.utf16().size(), 13);
    ASSERT_EQ(str2.utf16().size(), 13);

    ASSERT_EQ(str1.utf8().size(), str2.utf8().size());
    ASSERT_EQ(str1.utf16().size(), str2.utf16().size());
  }
  {
    String str2;
    Span<const Byte> b = str2.bytes();
    ASSERT_EQ(b.size(), 0);
    ByteConstArrayView u = str2.utf8();
    ASSERT_EQ(u.size(), 0);
  }

  {
    String str3("TX");
    Span<const Byte> b = str3.bytes();
    ASSERT_EQ(b.size(), 2);
    ByteConstArrayView u = str3.utf8();
    ASSERT_EQ(u.size(), 3);
    ASSERT_EQ(u[2], 0);
  }
  {
    String str4("€");
    std::array<Byte, 3> ref_a{ 0xe2, 0x82, 0xac };
    Span<const Byte> ref_a_view{ ref_a };
    Span<const Byte> b = str4.bytes();
    ASSERT_EQ(b.size(), 3);
    ASSERT_EQ(b, ref_a_view);
    ByteConstArrayView u = str4.utf8();
    ASSERT_EQ(u.size(), 4);
    ASSERT_EQ(u[3], 0);
    for (Integer i = 0; i < 3; ++i) {
      ASSERT_EQ(u[i], ref_a[i]);
      ASSERT_EQ(b[i], ref_a[i]);
    }
  }
  {
    String x2 = "\xc3\xb1";
    _doConvertTest("X2", x2);
    String x3 = "\xe2\x82\xa1";
    _doConvertTest("X3", x3);
    String x4 = "\xf0\x90\x8c\xbc";
    _doConvertTest("X4", x4);
  }
}

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
