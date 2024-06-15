//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public struct Real3x3
  {  
    public static readonly Real3x3 Zero = new Real3x3(Real3.Zero,Real3.Zero,Real3.Zero);

    public Real3 x; //!< première composante du triplet
    public Real3 y; //!< deuxième composante du triplet
    public Real3 z; //!< troisième composante du triplet

    public Real3x3(Real3 _x,Real3 _y,Real3 _z)
    {
      x = _x;
      y = _y;
      z = _z;
    }
    public override string ToString()
    {
      return "("+x+","+y+","+z+")";
    }

    //public bool IsNearlyZero()
    //{
    //  return x.IsNearlyZero() && y.IsNearlyZero() && z.IsNearlyZero();
    //}

    //! Ajoute \a b au triplet
    //Real3x3& add(Real3x3 b) { x+=b.x; y+=b.y; z+=b.z; return (*this); }
    //! Soustrait \a b au triplet
    //Real3x3& sub(Real3x3 b) { x-=b.x; y-=b.y; z-=b.z; return (*this); }
    //! Multiple chaque composante du triplet par la composant correspondant de \a b
    //Real3x3& mul(Real3x3 b) { x*=b.x; y*=b.y; z*=b.z; return (*this); }
    //! Divise chaque composante du triplet par la composant correspondant de \a b
    //Real3x3& div(Real3x3 b) { x/=b.x; y/=b.y; z/=b.z; return (*this); }
    //! Ajoute \a b à chaque composante du triplet
    //Real3x3& addSame(Real3 b) { x+=b; y+=b; z+=b; return (*this); }
    //! Soustrait \a b à chaque composante du triplet
    //Real3x3& subSame(Real3 b) { x-=b; y-=b; z-=b; return (*this); }
    //! Multiplie chaque composante du triplet par \a b
    //Real3x3& mulSame(Real3 b) { x*=b; y*=b; z*=b; return (*this); }
    //! Divise chaque composante du triplet par \a b
    //Real3x3& divSame(Real3 b) { x/=b; y/=b; z/=b; return (*this); }
    //! Ajoute \a b au triplet.
    //Real3x3& operator+= (Real3x3 b) { return add(b); }
    //! Soustrait \a b au triplet
    //Real3x3& operator-= (Real3x3 b) { return sub(b); }
    //! Multiple chaque composante du triplet par la composant correspondant de \a b
    //Real3x3& operator*= (Real3x3 b) { return mul(b); }
    //! Multiple chaque composante de la matrice par le réel \a b
    //void operator*= (Real b) { x*=b; y*=b; z*=b; }
    //! Divise chaque composante du triplet par la composant correspondant de \a b
    //Real3x3& operator/= (Real3x3 b) { return div(b); }
    //! Divise chaque composante de la matrice par le réel \a b
    //void operator/= (Real b) { x/=b; y/=b; z/=b; }
    
    //! Créé un triplet qui vaut ce triplet ajouté à \a b
    public static Real3x3 operator+(Real3x3 a,Real3x3 b)
    {
      return new Real3x3(a.x+b.x,a.y+b.y,a.z+b.z);
    }

    //! Créé un triplet qui vaut \a b soustrait de ce triplet
    public static Real3x3 operator-(Real3x3 a,Real3x3 b)
    {
      return new Real3x3(a.x-b.x,a.y-b.y,a.z-b.z);
    }

    //! Créé un tenseur opposé au tenseur actuel
    public static Real3x3 operator-(Real3x3 a)
    {
      return new Real3x3(-a.x,-a.y,-a.z);
    }

    //public static bool operator==(Real3x3 a,Real3x3 b)
    //{
      //return  a.x==b.x && a.y==b.y && a.z==b.z;
    //}

    /*! \brief Multiplication par un scalaire. */
    public static Real3x3 operator*(Real sca,Real3x3 vec)
    {
      return new Real3x3(vec.x*sca,vec.y*sca,vec.z*sca);
    }

    /*! \brief Multiplication par un scalaire. */
    public static Real3x3 operator*(Real3x3 vec,Real sca)
    {
      return new Real3x3(vec.x*sca,vec.y*sca,vec.z*sca);
    }

    /*! \brief Division par un scalaire. */
    public static Real3x3 operator/(Real3x3 vec,Real sca)
    {
      return new Real3x3(vec.x/sca,vec.y/sca,vec.z/sca);
    }

    public static bool operator==(Real3x3 a,Real3x3 b)
    {
      return  a.x==b.x && a.y==b.y && a.z==b.z;
    }

    public static bool operator!=(Real3x3 a,Real3x3 b)
    {
      return  a.x!=b.x || a.y!=b.y || a.z!=b.z;
    }

    public override bool Equals(object oa)
    {
      if (!(oa is Real3x3))
        return false;
      Real3x3 a = (Real3x3)oa;
      return this==a;
    }

    public override int GetHashCode()
    {
      //TODO: faire une fonction de hashage un peu mieux...
      return x.GetHashCode() + y.GetHashCode() + z.GetHashCode();
    }

    public static bool operator<(Real3x3 v1,Real3x3 v2)
    {
      if (v1.x==v2.x){
        if (v1.y==v2.y)
          return v1.z<v2.z;
        else
          return v1.y<v2.y;
      }
      return (v1.x<v2.x);
    }

    public static bool operator>(Real3x3 v1,Real3x3 v2)
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