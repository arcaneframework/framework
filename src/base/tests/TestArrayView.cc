// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <gtest/gtest.h>

#include "arccore/base/ArrayView.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array3View,Misc)
{
  using namespace Arccore;
  int nb_x = 3;
  int nb_y = 4;
  int nb_z = 5;
  std::vector<Int32> buf(nb_x*nb_y*nb_z);
  for( size_t i=0, n=buf.size(); i<n; ++i )
    buf[i] = (Int32)(i+1);

  ConstArray3View<Int32> v(buf.data(),nb_x,nb_y,nb_z);
  Integer global_index = 0;
  for( Integer x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Integer y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Integer z=0, zn=v.dim3Size(); z<zn; ++z ){
        ++global_index;
        Int32 val1 = v[x][y][z];
        Int32 val2 = v.item(x,y,z);
        std::cout  << " V=" << val1 << " x=" << x << " y=" << y << " z=" << z << '\n';
        ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
        ASSERT_TRUE(val1==global_index) << "Bad value v1=" << val1 << " expected=" << global_index;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Array4View,Misc)
{
  using namespace Arccore;
  int nb_x = 2;
  int nb_y = 3;
  int nb_z = 4;
  int nb_a = 5;
  std::vector<Int32> buf(nb_x*nb_y*nb_z*nb_a);
  for( size_t i=0, n=buf.size(); i<n; ++i )
    buf[i] = (Int32)(i+1);

  ConstArray4View<Int32> v(buf.data(),nb_x,nb_y,nb_z,nb_a);
  Integer global_index = 0;
  for( Integer x=0, xn=v.dim1Size(); x<xn; ++x ){
    for( Integer y=0, yn=v.dim2Size(); y<yn; ++y ){
      for( Integer z=0, zn=v.dim3Size(); z<zn; ++z ){
        for( Integer a=0, an=v.dim4Size(); a<an; ++a ){
          ++global_index;
          Int32 val1 = v[x][y][z][a];
          Int32 val2 = v.item(x,y,z,a);
          std::cout << " V=" << val1 << " x=" << x << " y=" << y << " z=" << z << " a=" << a << '\n';
          ASSERT_TRUE(val1==val2) << "Difference values v1=" << val1 << " v2=" << val2;
          ASSERT_TRUE(val1==global_index) << "Bad value v1=" << val1 << " expected=" << global_index;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
