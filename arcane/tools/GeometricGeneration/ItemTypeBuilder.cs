using System;
using System.Collections.Generic;

namespace GeometricGeneration
{
  public class ItemTypeBuilder
  {
    List<Item> m_items_type = new List<Item> ();
    public IList<Item> Items { get { return m_items_type; } }

    public ItemTypeBuilder ()
    {
      m_items_type.Add (CreateVertex ());
      m_items_type.Add (CreateLine2 ());
      m_items_type.Add (CreateTriangle3 ());
      m_items_type.Add (CreateQuad4 ());
      m_items_type.Add (CreatePentagon5 ());
      m_items_type.Add (CreateHexagon6 ());
      m_items_type.Add (CreateTetraedron4 ());
      m_items_type.Add (CreatePyramid5 ());
      m_items_type.Add (CreatePentaedron6 ());
      m_items_type.Add (CreateHexaedron8 ());
      m_items_type.Add (CreateHeptaedron10 ());
      m_items_type.Add (CreateOctaedron12 ());
    }

    public void ComputeSubVolume ()
    {
      List<int> connectic = new List<int> ();
      foreach (Item it in m_items_type) {
        if (it.Type != GeomType.Hexaedron8 && it.Type != GeomType.Pyramid5)
          continue;
        for (int i = 0; i < it.NbNode; ++i) {
          int[] edge_index = it.NodesEdges [i];
          int nb_edge = edge_index.Length;
          if (nb_edge == 3) {
            // Cas classique 3D
            Edge e0 = it.Edges [edge_index [0]];
            Edge e1 = it.Edges [edge_index [1]];
            Edge e2 = it.Edges [edge_index [2]];
            Face f0 = it.FindFace (e0, e1);
            Face f1 = it.FindFace (e1, e2);
            Face f2 = it.FindFace (e2, e0);
            Console.WriteLine ("I={0} Z=0 E={1} F={2}", i, e0.Id, f0.Id);
            Console.WriteLine ("I={0} Z=1 E={1} F={2}", i, e1.Id, f1.Id);
            Console.WriteLine ("I={0} Z=2 E={1} F={2}", i, e2.Id, f2.Id);
            Console.WriteLine ("node{0} edge{1} face{2} edge{3} edge{4} face{5} center face{6}",
              i, e0.Id, f0.Id, e1.Id, e2.Id, f2.Id, f1.Id
            );
          } else if (nb_edge == 4) {
            // Cas pyramide noeud sommet
            Edge e0 = it.Edges [edge_index [0]];
            Edge e1 = it.Edges [edge_index [1]];
            Edge e2 = it.Edges [edge_index [2]];
            Edge e3 = it.Edges [edge_index [3]];
            Face f0 = it.FindFace (e0, e3);
            Face f1 = it.FindFace (e1, e0);
            Face f2 = it.FindFace (e2, e1);
            Face f3 = it.FindFace (e3, e2);
            Console.WriteLine ("node4x4 edge{0} face{1} center face{2}", e0.Id, f0.Id, f1.Id);
            Console.WriteLine ("node4x4 edge{0} face{1} center face{2}", e0.Id, f1.Id, f2.Id);
            Console.WriteLine ("node4x4 edge{0} face{1} center face{2}", e0.Id, f2.Id, f3.Id);
            Console.WriteLine ("node4x4 edge{0} face{1} center face{2}", e0.Id, f3.Id, f0.Id);
          }
        }
      }
    }



    public static Item CreateVertex ()
    {
      Item it = new Item (1, GeomType.Vertex, "Vertex", 1, null, null);
      return it;
    }

    public static Item CreateLine2 ()
    {
      Item it = new Item (2, GeomType.Line2, "Line", 1, null, null);
      return it;
    }

    public static Item CreateTriangle3 ()
    {
      return Item.Create2DType (3, GeomType.Triangle3, "Triangle");
    }

    public static Item CreateQuad4 ()
    {
      return Item.Create2DType (4, GeomType.Quad4, "Quad");
    }

    public static Item CreatePentagon5 ()
    {
      return Item.Create2DType (5, GeomType.Pentagon5, "Pentagon");
    }

    public static Item CreateHexagon6 ()
    {
      return Item.Create2DType (6, GeomType.Hexagon6, "Hexagon");
    }

    public static Item CreateTetraedron4 ()
    {
      Face[] faces = {
        new Face (0, 2, 1),
        new Face (0, 3, 2),
        new Face (0, 1, 3),
        new Face (1, 2, 3)
      };

      Edge[] edges = {
        new Edge (0, 1),
        new Edge (1, 2),
        new Edge (2, 0),
        new Edge (0, 3),
        new Edge (1, 3),
        new Edge (2, 3)
      };
      int[][] node_edges = {
        new int[]{ 0, 2, 3 },
        new int[]{ 1, 0, 4 },
        new int[]{ 2, 1, 5 },
        new int[]{ 3, 5, 4 },
      };
      Item it = new Item (4, GeomType.Tetraedron4, "Tetra", 3, edges, faces);
      return it;
    }
    // La pyramide est considérée comme un hexaèdre dégénéré
    // pour les arêtes
    // La dégénérescence est au sommet de la pyramide. Ce sommet
    // est donc considéré 4 fois pour les arêtes.
    public static Item CreatePyramid5 ()
    {
      Face[] faces = {
        new Face(0,3,2,1),
        new Face(0,4,3),
        new Face(0,1,4),
        new Face(1,2,4),
        new Face(2,3,4)
      };

      Edge[] edges =
      {
        new Edge(0,1),
        new Edge(1,2),
        new Edge(2,3),
        new Edge(3,0),
        new Edge(0,4),
        new Edge(1,4),
        new Edge(2,4),
        new Edge(3,4),
        new Edge(4,4),
        new Edge(4,4),
        new Edge(4,4),
        new Edge(4,4)
      };
      int[][] node_edges =
      {
        new int[]{ 0, 3, 4 },
        new int[]{ 1, 0, 5 },
        new int[]{ 2, 1, 6 },
        new int[]{ 3, 2, 7 },

        // Special pyramide
        new int[]{ 4, 5, 6, 7 },
      };
      Item it = new Item(5,GeomType.Pyramid5,"Pyramid",3,edges,faces);
      it.NodesEdges = node_edges;
      return it;
    }

    public static Item CreatePentaedron6()
    {
      Face[] faces = 
      {
        new Face(0,2,1),
        new Face(0,3,5,2),
        new Face(0,1,4,3),
        new Face(3,4,5),
        new Face(1,2,5,4)
      };

      Edge[] edges =
      {
        new Edge(0,1),
        new Edge(1,2),
        new Edge(2,0),
        new Edge(0,3),
        new Edge(1,4),
        new Edge(2,5),
        new Edge(3,4),
        new Edge(4,5),
        new Edge(5,3)
      };
      int[][] node_edges =
      {
        new int[]{ 0, 2, 3 },
        new int[]{ 1, 0, 4 },
        new int[]{ 2, 1, 5 },
        new int[]{ 8, 6, 3 },
        new int[]{ 6, 7, 4 },
        new int[]{ 7, 8, 5 },
      };
      Item it = new Item(6,GeomType.Pentaedron6,"Penta",3,edges,faces);
      return it;
    }

    public static Item CreateHexaedron8()
    {
      Face[] faces = 
      {
        new Face(0,3,2,1),
        new Face(0,4,7,3),
        new Face(0,1,5,4),
        new Face(4,5,6,7),
        new Face(1,2,6,5),
        new Face(2,3,7,6)
      };

      Edge[] edges =
      {
        new Edge(0,1),
        new Edge(1,2),
        new Edge(2,3),
        new Edge(3,0),
        new Edge(0,4),
        new Edge(1,5),
        new Edge(2,6),
        new Edge(3,7),
        new Edge(4,5),
        new Edge(5,6),
        new Edge(6,7),
        new Edge(7,4)
      };
      Item it = new Item(8,GeomType.Hexaedron8,"Hexa",3,edges,faces);

      Real3[] coords =
      {
        new Real3(0.0,0.0,0.0),
        new Real3(1.0,0.0,0.0),
        new Real3(1.0,1.0,0.0),
        new Real3(0.0,1.0,0.0),
        new Real3(0.0,0.0,1.0),
        new Real3(1.0,0.0,1.0),
        new Real3(1.0,1.0,1.0),
        new Real3(0.0,1.0,1.0)
      };

      it.Coords = coords;

      int[][] node_edges =
      {
        new int[]{ 0,  3,  4 },
        new int[]{ 1,  0,  5 },
        new int[]{ 2,  1,  6 },
        new int[]{ 3,  2,  7 },
        new int[]{ 11, 8,  4 },
        new int[]{ 8,  9,  5 },
        new int[]{ 9,  10, 6 },
        new int[]{ 10, 11, 7 },
      };
      it.NodesEdges = node_edges;
      return it;
    }

    public static Item CreateHeptaedron10()
    {
      Face[] faces = 
      {
        new Face(0,4,3,2,1),
        new Face(5,6,7,8,9),
        new Face(0,1,6,5),
        new Face(1,2,7,6),
        new Face(2,3,8,7),
        new Face(3,4,9,8)
      };

      Edge[] edges =
      {
        new Edge(0,1),
        new Edge(1,2),
        new Edge(2,3),
        new Edge(3,4),
        new Edge(4,0),
        new Edge(5,6),
        new Edge(6,7),
        new Edge(7,8),
        new Edge(8,9),
        new Edge(9,5),
        new Edge(0,5),
        new Edge(1,6),
        new Edge(2,7),
        new Edge(3,8),
        new Edge(4,9)
      };
      Item it = new Item(10,GeomType.Heptaedron10,"Wedge7",3,edges,faces);
      return it;
    }

    public static Item CreateOctaedron12()
    {
      Face[] faces = 
      {
        new Face(0,5,4,3,2,1),
        new Face(6,7,8,9,10,11),
        new Face(0,1,7,6),
        new Face(1,2,8,7),
        new Face(2,3,9,8),
        new Face(3,4,10,9),
        new Face(4,5,11,10),
        new Face(5,0,6,11)
      };

      Edge[] edges =
      {
        new Edge(0,1),
        new Edge(1,2),
        new Edge(2,3),
        new Edge(3,4),
        new Edge(4,5),
        new Edge(5,0),
        new Edge(6,7),
        new Edge(7,8),
        new Edge(8,9),
        new Edge(9,10),
        new Edge(10,11),
        new Edge(11,6),
        new Edge(0,6),
        new Edge(1,7),
        new Edge(2,8),
        new Edge(3,9),
        new Edge(4,10),
        new Edge(5,11)
      };
      Item it = new Item(12,GeomType.Octaedron12,"Wedge8",3,edges,faces);
      return it;
    }

  }
}

