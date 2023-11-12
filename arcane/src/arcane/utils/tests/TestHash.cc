// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/SHA3HashAlgorithm.h"
#include "arcane/utils/SHA1HashAlgorithm.h"
#include "arcane/utils/Ref.h"

#include <gtest/gtest.h>

#include <array>

namespace Arcane
{
//extern "C++" ARCANE_UTILS_EXPORT void
//_computeSHA3_256Hash(Span<const std::byte> bytes, SmallSpan<std::byte> result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

struct TestInfo
{
  String data_value;
  String expected_hash;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _testHash(IHashAlgorithm& algo, SmallSpan<TestInfo> values_to_test)
{
  ByteUniqueArray output1;
  ByteUniqueArray output2;
  ByteUniqueArray output3;
  UniqueArray<std::byte> output1_incremental;

  Ref<IHashAlgorithmContext> context;
  bool has_context = false;
  if (algo.hasCreateContext()){
    context = algo.createContext();
    has_context = true;
  }

  // TODO: ajouter test ou ne fait clear() entre les appels
  for (const TestInfo& ti : values_to_test) {
    Span<const Byte> str_bytes(ti.data_value.bytes());
    {
      Span<const std::byte> input1(asBytes(str_bytes));
      output1.clear();
      algo.computeHash64(input1, output1);
      if (has_context){
        output1_incremental.clear();
        context->reset();
        // Fais l'appel en deux fois.
        auto span1 = input1.subspan(0,input1.size()/2);
        auto span2 = input1.subspan(input1.size()/2,input1.size());
        context->updateHash(span1);
        context->updateHash(span2);
        HashAlgorithmValue value;
        context->computeHashValue(value);
        output1_incremental = value.bytes();
      }
    }
    {
      Span<const Byte> input2(str_bytes);
      output2.clear();
      algo.computeHash64(input2, output2);
    }
    {
      ByteConstArrayView input3(str_bytes.constSmallView());
      output3.clear();
      algo.computeHash(input3, output3);
    }
    String hash1 = Convert::toHexaString(output1);
    String hash2 = Convert::toHexaString(output2);
    String hash3 = Convert::toHexaString(output3);
    String hash1_incremental = Convert::toHexaString(output1_incremental);
    String expected_hash = ti.expected_hash.lower();
    std::cout << "REF=" << expected_hash << "\n";
    std::cout << "HASH1=" << hash1 << "\n";
    std::cout << "HASH2=" << hash2 << "\n";
    std::cout << "HASH3=" << hash3 << "\n";
    std::cout << "HASH1_INCREMENTAL=" << hash1_incremental << "\n";
    ASSERT_EQ(hash1, expected_hash);
    if (has_context){
      ASSERT_EQ(hash1_incremental, expected_hash);
    }
    ASSERT_EQ(hash2, expected_hash);
    ASSERT_EQ(hash3, expected_hash);
  }

  {
    // Teste un gros tableau
    const Int32 nb_byte = 100000;
    UniqueArray<std::byte> bytes(nb_byte);
    for (Int32 i = 0; i < nb_byte; ++i) {
      bytes[i] = std::byte(i % 127);
    }
    // Boucle si on veut tester les performances
    //(dans ce cas augmenter le nombre d'itérations)
    for (int j = 0; j < 2; ++j) {
      output1.clear();
      algo.computeHash64(bytes, output1);
    }
    String hash_big = Convert::toHexaString(output1);
    std::cout << "HASH_BIG=" << hash_big << "\n";
    if (has_context){
      context->reset();
      Int32 to_add = 549;
      Int32 nb_iter = 0;
      for( Int32 i=0; i<nb_byte; i+=to_add ){
        ++nb_iter;
        to_add += 15;
        context->updateHash(bytes.span().subspan(i,to_add));
      }
      HashAlgorithmValue value;
      context->computeHashValue(value);
      output1_incremental = value.bytes();
      String hash_big_incremental = Convert::toHexaString(output1_incremental);
      std::cout << "nb_iter=" << nb_iter << "\n";
      std::cout << "HASH_BIG_INCREMENTAL=" << hash_big_incremental << "\n";
      ASSERT_EQ(hash_big_incremental, hash_big);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, SHA3_224)
{
  std::cout << "TEST_SHA3-224\n";

  std::array<TestInfo, 8> values_to_test = {
    { { "", "6B4E03423667DBB73B6E15454F0EB1ABD4597F9A1B078E3F5B5A6BC7" },
      { "a", "9E86FF69557CA95F405F081269685B38E3A819B309EE942F482B6A8B" },
      { "abc", "E642824C3F8CF24AD09234EE7D3C766FC9A3A5168D0C94AD73B46FDF" },
      { "message digest", "18768BB4C48EB7FC88E5DDB17EFCF2964ABD7798A39D86A4B4A1E4C8" },
      { "abcdefghijklmnopqrstuvwxyz", "5CDECA81E123F87CAD96B9CBA999F16F6D41549608D4E0F4681B8239" },
      { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", "A67C289B8250A6F437A20137985D605589A8C163D45261B15419556E" },
      { "12345678901234567890123456789012345678901234567890123456789012345678901234567890", "0526898E185869F91B3E2A76DD72A15DC6940A67C8164A044CD25CC8" },
      { "The quick brown fox jumps over the lazy dog", "D15DADCEAA4D5D7BB3B48F446421D542E08AD8887305E28D58335795" } }
  };

  SHA3_224HashAlgorithm sha3;
  _testHash(sha3, SmallSpan<TestInfo>(values_to_test));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, SHA3_256)
{
  std::cout << "TEST_SHA3-256\n";

  std::array<TestInfo, 8> values_to_test = {
    { { "",
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
        "69070dda01975c8c120c3aada1b282394e7f032fa9cf32f4cb2259a0897dfc04" } }
  };

  SHA3_256HashAlgorithm sha3;
  _testHash(sha3, SmallSpan<TestInfo>(values_to_test));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, SHA3_384)
{
  std::cout << "TEST_SHA3-384\n";

  std::array<TestInfo, 8> values_to_test = {
    { { "", "0C63A75B845E4F7D01107D852E4C2485C51A50AAAA94FC61995E71BBEE983A2AC3713831264ADB47FB6BD1E058D5F004" },
      { "a", "1815F774F320491B48569EFEC794D249EEB59AAE46D22BF77DAFE25C5EDC28D7EA44F93EE1234AA88F61C91912A4CCD9" },
      { "abc", "EC01498288516FC926459F58E2C6AD8DF9B473CB0FC08C2596DA7CF0E49BE4B298D88CEA927AC7F539F1EDF228376D25" },
      { "message digest",
        "D9519709F44AF73E2C8E291109A979DE3D61DC02BF69DEF7FBFFDFFFE662751513F19AD57E17D4B93BA1E484FC1980D5" },
      { "abcdefghijklmnopqrstuvwxyz",
        "FED399D2217AAF4C717AD0C5102C15589E1C990CC2B9A5029056A7F7485888D6AB65DB2370077A5CADB53FC9280D278F" },
      { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "D5B972302F5080D0830E0DE7B6B2CF383665A008F4C4F386A61112652C742D20CB45AA51BD4F542FC733E2719E999291" },
      { "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "3C213A17F514638ACB3BF17F109F3E24C16F9F14F085B52A2F2B81ADC0DB83DF1A58DB2CE013191B8BA72D8FAE7E2A5E" },
      { "The quick brown fox jumps over the lazy dog",
        "7063465E08A93BCE31CD89D2E3CA8F602498696E253592ED26F07BF7E703CF328581E1471A7BA7AB119B1A9EBDF8BE41" } }
  };

  SHA3_384HashAlgorithm sha3;
  _testHash(sha3, SmallSpan<TestInfo>(values_to_test));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, SHA3_512)
{
  std::cout << "TEST_SHA3-512\n";

  std::array<TestInfo, 8> values_to_test = {
    { { "", "A69F73CCA23A9AC5C8B567DC185A756E97C982164FE25859E0D1DCC1475C80A615B2123AF1F5F94C11E3E9402C3AC558F500199D95B6D3E301758586281DCD26" },
      { "a", "697F2D856172CB8309D6B8B97DAC4DE344B549D4DEE61EDFB4962D8698B7FA803F4F93FF24393586E28B5B957AC3D1D369420CE53332712F997BD336D09AB02A" },
      { "abc", "B751850B1A57168A5693CD924B6B096E08F621827444F70D884F5D0240D2712E10E116E9192AF3C91A7EC57647E3934057340B4CF408D5A56592F8274EEC53F0" },
      { "message digest",
        "3444E155881FA15511F57726C7D7CFE80302A7433067B29D59A71415CA9DD141AC892D310BC4D78128C98FDA839D18D7F0556F2FE7ACB3C0CDA4BFF3A25F5F59" },
      { "abcdefghijklmnopqrstuvwxyz",
        "AF328D17FA28753A3C9F5CB72E376B90440B96F0289E5703B729324A975AB384EDA565FC92AADED143669900D761861687ACDC0A5FFA358BD0571AAAD80ACA68" },
      { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "D1DB17B4745B255E5EB159F66593CC9C143850979FC7A3951796ABA80165AAB536B46174CE19E3F707F0E5C6487F5F03084BC0EC9461691EF20113E42AD28163" },
      { "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "9524B9A5536B91069526B4F6196B7E9475B4DA69E01F0C855797F224CD7335DDB286FD99B9B32FFE33B59AD424CC1744F6EB59137F5FB8601932E8A8AF0AE930" },
      { "The quick brown fox jumps over the lazy dog",
        "01DEDD5DE4EF14642445BA5F5B97C15E47B9AD931326E4B0727CD94CEFC44FFF23F07BF543139939B49128CAF436DC1BDEE54FCB24023A08D9403F9B4BF0D450" } }
  };

  SHA3_512HashAlgorithm sha3;
  _testHash(sha3, SmallSpan<TestInfo>(values_to_test));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, SHA1)
{
  std::cout << "TEST_SHA1\n";

  std::array<TestInfo, 3> values_to_test = {
    { { "", "da39a3ee5e6b4b0d3255bfef95601890afd80709" },
      { "The quick brown fox jumps over the lazy cog", "de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3" },
      { "The quick brown fox jumps over the lazy dog", "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12" } }
  };

  SHA1HashAlgorithm sha1;
  _testHash(sha1, SmallSpan<TestInfo>(values_to_test));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Hash, MD5)
{
  std::cout << "MD5 TEST\n";

  std::array<TestInfo, 4> values_to_test = {
    { { "",
        "d41d8cd98f00b204e9800998ecf8427e" },

      { "a",
        "0cc175b9c0f1b6a831c399e269772661" },

      { "The quick brown fox jumps over the lazy dog",
        "9e107d9d372bb6826bd81d3542a419d6" },

      { "The quick brown fox jumps over the lazy dog.",
        "e4d909c290d0fb1ca068ffaddf22cbd0" } }
  };

  Arcane::MD5HashAlgorithm md5;
  _testHash(md5, SmallSpan<TestInfo>(values_to_test));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
