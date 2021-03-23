using System;
using System.Collections.Generic;
using System.Text;

namespace Arcane.MatVec
{
  public interface IPreconditioner
  {
    void Apply(Vector out_vec, Vector vec);
  }
}
