using System;

namespace GeometricGeneration
{
  public struct Real3
  {
    public Real3 (double _x, double _y, double _z)
    {
      x = _x;
      y = _y;
      z = _z;
    }

    public double x;
    public double y;
    public double z;

    static public Real3 operator- (Real3 a, Real3 b)
    {
      return new Real3 (a.x - b.x, a.y - b.y, a.z - b.z);
    }

    public static Real3 Cross (Real3 u, Real3 v)
    {
      return new Real3 (u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x);
    }

    public override string ToString ()
    {
      return "(" + x + "," + y + "," + z + ")";
    }
    /*    public static bool operator==(Real3 a,Real3 b)
  {
    return (a.x==b.x && a.y==b.y && a.z==b.z);
  }
  public static bool operator!=(Real3 a,Real3 b)
  {
    return !(a==b);
  }*/
  }
}

