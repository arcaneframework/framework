using System;
using System.IO;
using System.Collections.Generic;
using GeometricGeneration;
using System.Text;

namespace GeometricGeneration
{
  public class Item
  {
    public IList<Face> Faces { get { return m_faces; } }

    private List<Face> m_faces = new List<Face> ();

    public IList<Edge> Edges { get { return m_edges; } }

    private List<Edge> m_edges = new List<Edge> ();

    public int NbNode { get { return m_nb_node; } }

    private int m_nb_node;

    public int NbEdge { get { return m_edges.Count; } }

    public int NbFace { get { return m_faces.Count; } }

    public GeomType Type { get { return m_type; } }

    private GeomType m_type;

    public string Name { get { return m_type.ToString (); } }

    public Real3[] Coords;
    public int[][] NodesEdges;

    public string SVCBasicName {
      get {
        if (m_dimension == 2)
          return "Quad4";
        if (m_dimension == 3)
          return "Hexaedron8";
        return "NOSVC";
      }
    }

    public string BasicName { get { return m_basic_name; } }

    string m_basic_name;

    public int Dimension { get { return m_dimension; } }

    int m_dimension;

    public Item (int nb_node, GeomType type, string basic_name, int dimension, Edge[] edges, Face[] faces)
    {
      m_basic_name = basic_name;
      m_nb_node = nb_node;
      m_type = type;
      m_dimension = dimension;
      if (faces != null)
        m_faces.AddRange (faces);
      if (edges != null)
        m_edges.AddRange (edges);

      for (int i = 0; i < m_edges.Count; ++i)
        m_edges [i].Id = i;
      for (int i = 0; i < m_faces.Count; ++i)
        m_faces [i].Id = i;
    }

    public Face FindFace (Edge e1, Edge e2)
    {
      foreach (Face f in Faces) {
        bool is_valid = f.HasNode (e1.FirstNode) && f.HasNode (e1.SecondNode) &&
                        f.HasNode (e2.FirstNode) && f.HasNode (e2.SecondNode);
        if (is_valid)
          return f;
      }
      return null;
    }

    public static Item Create2DType (int n, GeomType type, string basic_name)
    {
      Face[] faces = new Face[n];
      Edge[] edges = new Edge[n];
      for (int i = 0; i < n; ++i) {
        faces [i] = new Face (i, (i + 1) % n);
        edges [i] = new Edge (i, (i + 1) % n);
      }
      Item it = new Item (n, type, basic_name, 2, edges, faces);
      return it;
    }

    public void PrintType ()
    {
      List<int> connectic = new List<int> ();
      Console.WriteLine ("Infos for '{0}'", Name);
      for (int i = 0; i < NbNode; ++i) {
        connectic.Clear ();
        ComputeNodeEdgeConnectic (i, connectic);
        PrintConnectic ("EDGE CONNECTIC FOR " + i, connectic);
      }

      for (int i = 0; i < NbNode; ++i) {
        connectic.Clear ();
        this.ComputeNodeFaceConnectic (i, connectic);
        PrintConnectic ("FACE CONNECTIC FOR " + i, connectic);
      }

      for (int i = 0; i < NbNode; ++i) {
        connectic.Clear ();
        ComputeNodeNodeFromEdgeConnectic (i, connectic);
        PrintConnectic ("NODE CONNECTIC FOR " + i, connectic);
      }
    }

    public void ComputeNodeEdgeConnectic (int i, List<int> connectic)
    {
      foreach (Edge edge in Edges) {
        if (edge.FirstNode == i)
          connectic.Add (edge.Id);
        if (edge.SecondNode == i)
          connectic.Add (edge.Id);
      }
    }

    public void ComputeNodeNodeFromEdgeConnectic (int i, List<int> connectic)
    {
      foreach (Edge edge in Edges) {
        if (edge.FirstNode == i)
          connectic.Add (edge.SecondNode);
        if (edge.SecondNode == i)
          connectic.Add (edge.FirstNode);
      }
    }

    public void ComputeNodeFaceConnectic (int i, List<int> connectic)
    {
      foreach (Face face in Faces) {
        for (int z = 0; z < face.Nodes.Length; ++z)
          if (face.Nodes [z] == i)
            connectic.Add (face.Id);
      }
    }

    public void PrintConnectic (string message, List<int> connectic)
    {
      Console.Write (message + " : ");
      foreach (int z in connectic) {
        Console.Write (" {0}", z);
      }
      Console.WriteLine (".");
    }
    /*!
     * \brief Retourne une chaîne de caractères contenant une liste de coordonnées
     * comme argument.
     * 
     * Par exemple, si l'élément à 3 coordonnées, retourne: const Real3& a0, const Real3& a1, const Real3& a2
     */
    public string CoordsArgString()
    {
      StringBuilder sb = new StringBuilder ();
      for (int i = 0; i < NbNode; ++i) {
        if (i != 0)
          sb.Append (", ");
        sb.Append ("const Real3& a");
        sb.Append (i);
      }
      return sb.ToString ();

    }
  }
}
