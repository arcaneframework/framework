using System;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public interface ITraceMng
  {
    void PutTrace(string str,int type);
  }
}
