using System;
using System.Collections.Generic;
using System.Text;

namespace GeometricGenerationScriban
{
  /// <summary>
  /// Type of a geometric element (copy of Arcane::geometric::GeomType).
  /// NOTE: the identifiers must correspond to those of the enumeration
  /// GeomType of 'arcane/src/arcane/geometry/GeomType.h' and to the
  /// descriptors of 'GeometricGenerationCpp/GeomShapeInfos.h'; this is
  /// checked at startup by GeomShapeInfos.Check().
  /// </summary>
  public enum GeomType
  {
    Vertex,
    Line2,
    Triangle3,
    Quad4,
    Pentagon5,
    Hexagon6,
    Tetraedron4,
    Pyramid5,
    Pentaedron6,
    Hexaedron8,
    Heptaedron10,
    Octaedron12
  }

  /// <summary>
  /// Description of one geometric shape type.
  /// </summary>
  public sealed class GeomShapeInfo
  {
    public string Name { get; }
    public GeomType Type { get; }
    public int NbNode { get; }
    public int NbEdge { get; }
    public int NbFace { get; }
    public int Dimension { get; }
    public string BasicName { get; }
    //! Name of the element holding a control sub-volume ("Quad4" or "Hexaedron8")
    public string SvcName { get; }
    //! Arguments "const Real3& a0, const Real3& a1, ..." (NbNode arguments)
    public string CoordsArgs { get; }

    internal GeomShapeInfo(GeomType type, int nb_node, int nb_edge, int nb_face,
                           int dimension, string basic_name)
    {
      Name = type.ToString();
      Type = type;
      NbNode = nb_node;
      NbEdge = nb_edge;
      NbFace = nb_face;
      Dimension = dimension;
      BasicName = basic_name;
      SvcName = (dimension == 2) ? "Quad4" : "Hexaedron8";
      var args = new StringBuilder();
      for (int i = 0; i < nb_node; ++i) {
        if (i != 0)
          args.Append(", ");
        args.Append("const Real3& a").Append(i);
      }
      CoordsArgs = args.ToString();
    }
  }

  /// <summary>
  /// Static description of the geometric shape types.
  /// This is the data model of the generator.
  /// </summary>
  public static class GeomShapeInfos
  {
    static readonly GeomShapeInfo[] _items = new GeomShapeInfo[]{
      new GeomShapeInfo(GeomType.Vertex, 1, 0, 0, 1, "Vertex"),
      new GeomShapeInfo(GeomType.Line2, 2, 0, 0, 1, "Line"),
      new GeomShapeInfo(GeomType.Triangle3, 3, 3, 3, 2, "Triangle"),
      new GeomShapeInfo(GeomType.Quad4, 4, 4, 4, 2, "Quad"),
      new GeomShapeInfo(GeomType.Pentagon5, 5, 5, 5, 2, "Pentagon"),
      new GeomShapeInfo(GeomType.Hexagon6, 6, 6, 6, 2, "Hexagon"),
      new GeomShapeInfo(GeomType.Tetraedron4, 4, 6, 4, 3, "Tetra"),
      new GeomShapeInfo(GeomType.Pyramid5, 5, 12, 5, 3, "Pyramid"),
      new GeomShapeInfo(GeomType.Pentaedron6, 6, 9, 5, 3, "Penta"),
      new GeomShapeInfo(GeomType.Hexaedron8, 8, 12, 6, 3, "Hexa"),
      new GeomShapeInfo(GeomType.Heptaedron10, 10, 15, 6, 3, "Wedge7"),
      new GeomShapeInfo(GeomType.Octaedron12, 12, 18, 8, 3, "Wedge8")
    };

    public static IReadOnlyList<GeomShapeInfo> Items {
      get { return _items; }
    }

    public static IReadOnlyList<GeomShapeInfo> Items2DAnd3D {
      get {
        var result = new List<GeomShapeInfo>();
        foreach (var item in _items)
          if (item.Dimension == 2 || item.Dimension == 3)
            result.Add(item);
        return result;
      }
    }

    /// <summary>
    /// Checks that each GeomType enumerator has exactly one descriptor
    /// with the same identifier and the same value, and vice versa.
    /// </summary>
    public static void Check()
    {
      string[] names = Enum.GetNames(typeof(GeomType));
      if (names.Length != _items.Length)
        throw new InvalidOperationException("The number of GeomType enumerators does not match the number of descriptors");
      foreach (var name in names) {
        GeomShapeInfo found = null;
        foreach (var item in _items) {
          if (item.Name == name) {
            if (found != null)
              throw new InvalidOperationException("Duplicate descriptor for GeomType::" + name);
            found = item;
          }
        }
        if (found == null)
          throw new InvalidOperationException("Missing descriptor for GeomType::" + name);
        if (found.Type != (GeomType)Enum.Parse(typeof(GeomType), name))
          throw new InvalidOperationException("Mismatched value for descriptor GeomType::" + name);
      }
    }
  }
}
