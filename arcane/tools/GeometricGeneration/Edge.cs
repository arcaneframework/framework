using System;

namespace GeometricGeneration
{
  public class Edge
  {
    public int FirstNode;
    public int SecondNode;
    public int Id;

    public Edge (int n1, int n2)
    {
      FirstNode = n1;
      SecondNode = n2;
    }
  }
}

