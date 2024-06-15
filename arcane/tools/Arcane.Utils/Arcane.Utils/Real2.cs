//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public struct Real2
  {  
    public static readonly Real2 Zero = new Real2(0.0,0.0);

    public Real x; //!< première composante du couple
    public Real y; //!< deuxième composante du couple

    public Real2(Real _x,Real _y)
    {
      x = _x;
      y = _y;
    }
    public static Real2 operator-(Real2 u,Real2 v)
    {
      return new Real2(u.x-v.x,u.y-v.y);
    }
    public static Real2 operator+(Real2 u,Real2 v)
    {
      return new Real2(u.x+v.x,u.y+v.y);
    }
    public static Real2 operator*(double a,Real2 v)
    {
      return new Real2(a*v.x, a*v.y);
    }
    public static Real2 operator*(Real2 u,double b)
    {
      return new Real2(u.x*b, u.y*b);
    }
    public static Real2 operator/(Real2 u,double b)
    {
      return new Real2(u.x/b, u.y/b);
    }
    public Real Abs()
    {
      return Math.Sqrt(x*x+y*y);
    }
    public override string ToString()
    {
      return "("+x+","+y+")";
    }
    public void Normalize()
    {
      Real v = Abs();
      if (v!=0.0)
        this /= v;
    }
    public static bool operator<(Real2 v1,Real2 v2)
    {     
      if (v1.x==v2.x){
          return v1.y<v2.y;
      }
      return (v1.x<v2.x);
    }
   public static bool operator>(Real2 v1,Real2 v2)
    {     
      if (v1.x==v2.x){
          return v1.y>v2.y;
      }
      return (v1.x>v2.x);
    }
  }
}
