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
  public struct Real3
  {  
    public static readonly Real3 Zero = new Real3(0.0,0.0,0.0);

    public Real x; //!< première composante du triplet
    public Real y; //!< deuxième composante du triplet
    public Real z; //!< troisième composante du triplet

    public Real3(Real _x,Real _y,Real _z)
    {
      x = _x;
      y = _y;
      z = _z;
    }
    public static Real3 operator-(Real3 u)
    {
      return new Real3(-u.x,-u.y,-u.z);
    }
    public static Real3 operator-(Real3 u,Real3 v)
    {
      return new Real3(u.x-v.x,u.y-v.y,u.z-v.z);
    }
    public static Real3 operator+(Real3 u,Real3 v)
    {
      return new Real3(u.x+v.x,u.y+v.y,u.z+v.z);
    }
    public static Real3 operator+(Real3 u,Real v)
    {
      return new Real3(u.x+v,u.y+v,u.z+v);
    }
    public static Real3 operator*(double a,Real3 v)
    {
      return new Real3(a*v.x, a*v.y, a*v.z);
    }
    public static Real3 operator*(Real3 u,double b)
    {
      return new Real3(u.x*b, u.y*b, u.z*b);
    }
    public static Real3 operator/(Real3 u,double b)
    {
      return new Real3(u.x/b, u.y/b, u.z/b);
    }
    public Real Abs()
    {
      return Math.Sqrt(x*x+y*y+z*z);
    }
    public Real Abs2()
    {
      return x*x+y*y+z*z;
    }
    public void Normalize()
    {
      Real v = Abs();
      if (v!=0.0)
        this /= v;
    }
    public override string ToString()
    {
      return "("+x+","+y+","+z+")";
    }

    public static bool operator==(Real3 a,Real3 b)
    {
      return  a.x==b.x && a.y==b.y && a.z==b.z;
    }

    public override bool Equals(object oa)
    {
      if (!(oa is Real3))
        return false;
      Real3 a = (Real3)oa;
      return this==a;
    }

    public override int GetHashCode()
    {
      //TODO: faire une fonction de hashage un peu mieux...
      return x.GetHashCode() + y.GetHashCode() + z.GetHashCode();
    }

    public static bool operator!=(Real3 a,Real3 b)
    {
      return  a.x!=b.x || a.y!=b.y || a.z!=b.z;
    }

    public static bool operator<(Real3 v1,Real3 v2)
    {     
      if (v1.x==v2.x){
        if (v1.y==v2.y)
          return v1.z<v2.z;
        else
          return v1.y<v2.y;
      }
      return (v1.x<v2.x);
    }
   public static bool operator>(Real3 v1,Real3 v2)
    {     
      if (v1.x==v2.x){
        if (v1.y==v2.y)
          return v1.z>v2.z;
        else
          return v1.y>v2.y;
      }
      return (v1.x>v2.x);
    }
    }
}
