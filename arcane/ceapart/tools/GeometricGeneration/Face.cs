using System;

namespace GeometricGeneration
{
  public class Face
  {
    public int[] Nodes;

    public int NbNode { get { return Nodes.Length; } }

    public int Id;

    public Face (int n1, int n2, int n3, int n4, int n5, int n6)
    {
      Nodes = new int[]{ n1, n2, n3, n4, n5, n6 };
    }

    public Face (int n1, int n2, int n3, int n4, int n5)
    {
      Nodes = new int[]{ n1, n2, n3, n4, n5 };
    }

    public Face (int n1, int n2, int n3, int n4)
    {
      Nodes = new int[]{ n1, n2, n3, n4 };
    }

    public Face (int n1, int n2, int n3)
    {
      Nodes = new int[]{ n1, n2, n3 };
    }

    public Face (int n1, int n2)
    {
      Nodes = new int[]{ n1, n2 };
    }

    public bool HasNode (int index)
    {
      for (int i = 0; i < Nodes.Length; ++i) {
        if (Nodes [i] == index)
          return true;
      }
      return false;
    }
  }
}

