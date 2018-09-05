#include <gtest/gtest.h>

#include "arccore/base/String.h"
#include "arccore/base/TraceInfo.h"

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
