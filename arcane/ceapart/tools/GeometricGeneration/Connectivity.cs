using System;
using System.IO;
using System.Collections.Generic;
using GeometricGeneration;

namespace GeometricGeneration
{
  public class Connectivity
  {
    static public ICollection<Item> Items;
    static public ICollection<Item> Items2DAnd3D;

    static public void Create(ItemTypeBuilder builder)
    {
      List<Item> items = new List<Item> ();
      List<Item> items2dAnd3d = new List<Item> ();
      foreach (Item i in builder.Items) {
        items.Add (i);
        if (i.Dimension == 2 || i.Dimension == 3)
          items2dAnd3d.Add (i);
      }
      Items = items;
      Items2DAnd3D = items2dAnd3d;
    }
  }
}