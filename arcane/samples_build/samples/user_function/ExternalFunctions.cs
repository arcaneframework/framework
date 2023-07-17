using System;
using Arcane;
using Real = System.Double;

namespace UserFunctionSample
{
  public class CaseFunctions
  {
    Real3 m_origin;

    // Constructor to initialize instance
    public CaseFunctions()
    {
      m_origin = new Real3(0.2,0.3,0.0);
    }

    // User Function.
    public Real3 NodeVelocityFunc(Real global_time,Real3 position)
    {
      Real3 delta = position - m_origin;
      Real dx = delta.x;
      Real dy = delta.y;
      Real dz = delta.z;
      Real norm = 0.3 * System.Math.Sqrt(dx*dx+dy*dy+dz*dz);
      Real vx = global_time * norm * dx;
      Real vy = global_time * norm * dy;
      Real vz = global_time * norm * dz;
      return new Real3(vx,vy,vz);
    }
  }
}
