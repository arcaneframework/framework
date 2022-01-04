//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using Real = System.Double;

namespace Arcane
{
 static public class Math
 {
   public static Real3 VecMul(Real3 u,Real3 v)
   {
     return new Real3(
                      u.y * v.z  - u.z * v.y,
                      u.z * v.x  - u.x * v.z,
                      u.x * v.y  - u.y * v.x
                      );
   }
   public static Real Sqrt(Real v)
   {
     return System.Math.Sqrt(v);
   }

   public static Real Min(Real a,Real b)
   {
     return System.Math.Min(a,b);
   }

   public static Real Max(Real a,Real b)
   {
     return System.Math.Max(a,b);
   }
   
   public static Real Dot(Real3 a,Real3 b)
   {
     return a.x*b.x + a.y*b.y + a.z*b.z;
   }
 }
}
