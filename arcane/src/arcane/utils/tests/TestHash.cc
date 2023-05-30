// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/MD5HashAlgorithm.h"

#include <gtest/gtest.h>

#include <array>

namespace Arcane
{
extern "C++" ARCANE_UTILS_EXPORT void
_computeSHA3_256Hash(Span<const std::byte> bytes, SmallSpan<std::byte> result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

struct TestInfo
{
  String data_value;
  String expected_hash;
};

TEST(Hash, SHA3_256)
{
  std::array<std::byte, 32> hash_result_array;
  SmallSpan<std::byte> hash_result(hash_result_array);

  std::cout << "TEST_SHA3-256\n";

  TestInfo values_to_test[8] = {
    { "",
      "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a" },

    { "a",
      "80084bf2fba02475726feb2cab2d8215eab14bc6bdd8bfb2c8151257032ecd8b" },

    { "abc",
      "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532" },

    { "message digest",
      "edcdb2069366e75243860c18c3a11465eca34bce6143d30c8665cefcfd32bffd" },

    { "abcdefghijklmnopqrstuvwxyz",
      "7cab2dc765e21b241dbc1c255ce620b29f527c6d5e7f5f843e56288f0d707521" },

    { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
      "a79d6a9da47f04a3b9a9323ec9991f2105d4c78a7bc7beeb103855a7a11dfb9f" },

    { "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
      "293e5ce4ce54ee71990ab06e511b7ccd62722b1beb414f5ff65c8274e0f5be1d" },

    { "The quick brown fox jumps over the lazy dog",
      "69070dda01975c8c120c3aada1b282394e7f032fa9cf32f4cb2259a0897dfc04" }
  };

  for (const TestInfo& ti : values_to_test) {
    Span<const std::byte> bytes(asBytes(ti.data_value.bytes()));
    _computeSHA3_256Hash(bytes, hash_result);
    ByteConstArrayView v(32, (Byte*)hash_result.data());
    String ref_str = Convert::toHexaString(v);
    std::cout << "CUR V3=" << ref_str << "\n";
    std::cout << "REF V3=" << ti.expected_hash << "\n";
    ASSERT_EQ(ref_str, ti.expected_hash);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, MD5)
{
  std::cout << "MD5 TEST\n";
  Arcane::MD5HashAlgorithm md5;
  ByteUniqueArray output1;
  ByteUniqueArray output2;
  ByteUniqueArray output3;

  TestInfo values_to_test[4] = {
    { "",
      "d41d8cd98f00b204e9800998ecf8427e" },

    { "a",
      "0cc175b9c0f1b6a831c399e269772661" },

    { "The quick brown fox jumps over the lazy dog",
      "9e107d9d372bb6826bd81d3542a419d6" },

    { "The quick brown fox jumps over the lazy dog.",
      "e4d909c290d0fb1ca068ffaddf22cbd0" }
  };

  for (const TestInfo& ti : values_to_test) {
    Span<const Byte> str_bytes(ti.data_value.bytes());
    {
      Span<const std::byte> input1(asBytes(str_bytes));
      output1.clear();
      md5.computeHash64(input1, output1);
    }
    {
      Span<const Byte> input2(str_bytes);
      output2.clear();
      md5.computeHash64(input2, output2);
    }
    {
      ByteConstArrayView input3(str_bytes.constSmallView());
      output3.clear();
      md5.computeHash(input3, output3);
    }
    String hash1 = Convert::toHexaString(output1);
    String hash2 = Convert::toHexaString(output2);
    String hash3 = Convert::toHexaString(output3);
    std::cout << "REF=" << ti.expected_hash << "\n";
    std::cout << "HASH1=" << hash1 << "\n";
    std::cout << "HASH2=" << hash2 << "\n";
    std::cout << "HASH3=" << hash3 << "\n";
    ASSERT_EQ(hash1, ti.expected_hash);
    ASSERT_EQ(hash2, ti.expected_hash);
    ASSERT_EQ(hash3, ti.expected_hash);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
