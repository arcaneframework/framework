using System;
using Arcane;
using Real = System.Double;
#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Math = Arcane.Math;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

public class MyClass : IDisposable
{
  public static int NB;
  public int[] z;
  public GCHandle m_handle;
  public IntPtr m_ptr;
  public MyClass()
  {
    z = new int[100];
    ++NB;
    NB += z[10];
    m_handle = GCHandle.Alloc(z,GCHandleType.Pinned);
    m_ptr = Marshal.UnsafeAddrOfPinnedArrayElement(z,0);
    //m_view = new Int32ArrayView((Int32*)ptr,array.Length);
  }
  //~MyClass()
  //{
  //m_handle.Free();
  //Dispose();
  //}

  public void Dispose()
  {
    Dispose(true);
    GC.SuppressFinalize(this);
    //m_handle = null;
    //Console.WriteLine("EXPLICIT DISPOSE MY CLASS");
  }
  public void Dispose(bool disposing)
  {
    //if (m_handle!=IntPtr.Zero)
    m_handle.Free();
    //m_handle = IntPtr.Zero;
    //GC.SuppressFinalize(this);
    //m_handle = null;
    //Console.WriteLine("DISPOSE MY CLASS");
  }
}

public class MyStruct : IDisposable
{
  public int z;
  public MyClass c;
  public void Dispose()
  {
    Dispose(true);
    GC.SuppressFinalize(this);
  }
  public void Dispose(bool d)
  {
    Console.WriteLine("DISPOSE MY STRUCT");
  }
}

[Arcane.Service("MeshModification",typeof(Arcane.IUnitTest))]
public class MeshModificationService
//: Arcane.IUnitTest_WrapperService
: ArcaneMeshModificationObject
{
  TraceAccessor m_trace_accessor;
  public TraceAccessor Trace { get { return m_trace_accessor; } }

  public MeshModificationService(ServiceBuildInfo sbi) : base(sbi)
  {
    Console.WriteLine("CREATE MeshModificationServiceInstance");

    m_trace_accessor = new TraceAccessor(sbi.SubDomain().TraceMng());
  }
  public override void InitializeTest()
  {
  }
  public override void FinalizeTest()
  {
  }
  void _AllocTest()
  {
    for( int j=0; j<1; ++j ){
      MyStruct ms = new MyStruct();
      for(int i=0; i<100000; ++i ){
        ms.c = new MyClass();
        ms.c.Dispose();
      }
      ms.Dispose();
      GC.Collect();
    }
  }
  public override void ExecuteTest()
  {
    Console.WriteLine("MeshModificationService executeTest()");
    _PrintCells();
    _AllocTest();
    _RemoveCells();
    _RefineCells();
    GC.Collect();
    for( int i=0; i<30; ++i ){
      _CheckGC();
      GC.Collect();
    }
  }

  void _PrintCells()
  {
    IItemFamily cell_family = Mesh().CellFamily();
    CellGroup all_cells = cell_family.AllItems().CellGroup();
    ItemVectorView all_cells_view = all_cells.View();
    ItemVectorView<Cell> all_cells_view2 = new ItemVectorView<Cell>(all_cells_view);
    Console.WriteLine("PrintCells");
    foreach(Cell c in all_cells_view2){
      int nb_node = c.NbNode;
      Console.WriteLine("Cell: c={0}",c.LocalId, nb_node);
      for( int i=0; i<nb_node; ++i )
        Console.WriteLine("Node lid={0}",c.NodeLocalId(i));
      foreach(Node n in c.Nodes)
        Console.WriteLine("Node: n={0}",n.LocalId);
    }
  }

  void _RemoveCells()
  {
    //Console.WriteLine("IT IS MY TEST! {0} {1}",ms.z,MyClass.NB);
    IItemFamily cell_family = Mesh().CellFamily();
    CellGroup all_cells = cell_family.AllItems().CellGroup();

    ItemVectorView cell_view = all_cells.View();
    int n = cell_view.Size;
    Console.WriteLine($"RemoveCells n={n}");
    for( int i=0; i<n; ++i ){
      Console.WriteLine("CELL_VIEW i={0} id={1}",i,cell_view.Indexes[i]);
    }
    foreach(Item ie in cell_view){
      Console.WriteLine("CELL_VIEW2 ie={0}",ie.LocalId);
    }
    ItemGroup all_items = all_cells;
    foreach(Item ie in all_items){
      Console.WriteLine("CELL_VIEW3 ie={0}",ie.LocalId);
    }

    //Int32Array to_destroy_array = new Int32Array();
    //Int32Array.Wrapper to_destroy_array =
    Int32Array to_destroy_array = new Int32Array();
    Int32 modulo_to_remove = Options.RemoveFraction.Value();
    foreach(Cell c in all_cells){
      Console.WriteLine("Cell={0} lid={1}",c.UniqueId,c.LocalId);
      if ((c.LocalId % modulo_to_remove)==0)
        to_destroy_array.Add(c.LocalId);
    }
    IMeshModifier modifier = Mesh().Modifier();
    //int[] values = to_destroy_array.ToArray();
    //Int32ArrayView.Wrapper w = new Int32ArrayView.Wrapper(to_destroy_array.ToArray());
    //Int32ArrayView view = w.View;
    //for( int i=0; i<view.Length; ++i )
    //Console.WriteLine("VIEW I={0} V={1}",i,view[i]);
    modifier.RemoveCells(to_destroy_array.ConstView);
    modifier.EndUpdate();
    //ms.Dispose();
  }

  /*!
   * \brief Raffine certaines mailles en pyramide.
   *
   * Prend un hexaèdre sur trois:
   * - créé un noeud au centre de cet hexaèdre
   * - detache l'héxaèdre
   * - créé une pyramide (avec le uniqueId() de l'hexaèdre d'origine), en prenant comme base
   * la première face de l'hexaèdre et comme sommet de la pyramide le noeud créé.
   * - supprime les mailles détachées.
   */
  void _RefineCells()
  {
    IMesh mesh = Mesh();
    Int64 max_node_uid = _SearchMaxUniqueId(mesh.AllNodes());
    Int64 max_cell_uid = _SearchMaxUniqueId(mesh.AllCells());
    Trace.Info("MAX UID NODE={0} CELL={1}",max_node_uid,max_cell_uid);

    CellGroup all_cells = Mesh().AllCells();
    int index = 0;
    Integer nb_cell_to_add = 0;
    Int32Array cells_to_detach = new Int32Array();
    Int64Array to_add_cells = new Int64Array();
    Int64Array nodes_to_add = new Int64Array();
    List<Real3> nodes_to_add_coords = new List<Real3>();
    VariableNodeReal3 nodes_coords = mesh.NodesCoordinates();
    foreach(Cell c in all_cells){
      //TODO: tester si la maille est un hexaedre
      //if (c.
      ++index;
      if ((index % 3) == 0){
        cells_to_detach.Add(c.LocalId);

        to_add_cells.Add(8); // Pour une pyramide
        //to_add_cells.Add(max_cell_uid + index); // Pour le uid
        to_add_cells.Add(c.UniqueId + max_cell_uid); // Pour le uid, reutilise celui de la maille supprimée
        //to_add_cells.Add(c.UniqueId); // Pour le uid, reutilise celui de la maille supprimée
        to_add_cells.Add(c.Node(0).UniqueId);
        to_add_cells.Add(c.Node(1).UniqueId);
        to_add_cells.Add(c.Node(2).UniqueId);
        to_add_cells.Add(c.Node(3).UniqueId);
        Real3 center = new Real3();
        foreach(Node n in c.Nodes){
          center += nodes_coords[n];
          Trace.Info("ADD CENTER {0} node_uid={1} cell={2}",nodes_coords[n],n.UniqueId,c.UniqueId);
          Trace.Info("node_nb_cell={0}",n.NbCell);
          foreach(Cell c2 in n.Cells){
            Trace.Info("SUB NODE {0}",c2.UniqueId);
          }
        }
        center /= (Real)c.NbNode;
        Int64 node_uid = max_node_uid + index;
        Int64 cell_uid = max_cell_uid + index;
        nodes_to_add.Add(node_uid);
        nodes_to_add_coords.Add(center);
        to_add_cells.Add(node_uid);
        Trace.Info("WANT ADD NODE UID={0} COORD={1} CELL_UID={2}",node_uid,center,cell_uid);
        ++nb_cell_to_add;
      }
    }

    IMeshModifier modifier = mesh.Modifier();
    Integer nb_node_added = nodes_to_add.Length;
    Int32Array new_nodes_local_id = new Int32Array(nb_node_added);

    modifier.AddNodes(nodes_to_add.ConstView,new_nodes_local_id.View);
    Mesh().NodeFamily().EndUpdate();
    Trace.Info("NODES ADDED = {0}",nb_node_added);
    ItemInternalArrayView new_nodes_legacy = Mesh().NodeFamily().ItemsInternal();
    ItemInfoListView new_nodes = Mesh().NodeFamily().ItemInfoListView();
    Trace.Info("NB TOTAL NODE = {0}",new_nodes_legacy.Size);
    for(int i=0; i<nb_node_added; ++i){
      Int32 new_local_id = new_nodes_local_id[i];
      Item new_node = new_nodes[new_local_id];
      Trace.Info("NEW LOCAL ID={0} Coord={1} UID={2} FromItem={3}",
                 new_local_id, nodes_to_add_coords[i],new_node.UniqueId,new_node.LocalId);
      nodes_coords[new_nodes[new_local_id]] = nodes_to_add_coords[i];
      if (new_node.LocalId!=new_nodes_legacy[new_local_id].LocalId)
        throw new ApplicationException("Nodes are different (1)");
      if (new_node.LocalId!=new_local_id)
        throw new ApplicationException("Nodes are different (2)");
    }

    Trace.Info("NB CELL TO ADD = {0}",nb_cell_to_add);
    Trace.Warning("TEST WARNING");
    Int64ConstArrayView uid_view = to_add_cells.ConstView;
    for( Integer i=0; i<uid_view.Size; ++i ){
      Console.Write(" ");
      Console.Write(uid_view[i]);
    }
    Console.WriteLine(".");

    // Avant d'ajouter les nouvelles mailles, il faut détacher les anciennes
    modifier.DetachCells(cells_to_detach.ConstView);
    modifier.AddCells(nb_cell_to_add,to_add_cells.ConstView);
    modifier.RemoveDetachedCells(cells_to_detach.ConstView);
    modifier.EndUpdate();
    // Pour l'instant indispensable. Il faudra le supprimer par la suite
    //nodes_coords.Dispose();
  }

  void _CheckGC()
  {
    CellGroup all_cells = Mesh().AllCells();
    VariableNodeReal3 nodes_coords = Mesh().NodesCoordinates();
    double total = 0.0;
    foreach(Cell c in all_cells){
      Real3 center = new Real3();
      foreach(Node n in c.Nodes){
        center += nodes_coords[n];
      }
      total += center.Abs2();
    }
    Trace.Info("TOTAL={0}",total);
  }

  Int64 _SearchMaxUniqueId(ItemGroup group)
  {
    Int64 max_uid = 0;
    foreach(Item i in group){
      //Console.WriteLine("IT IS MY TEST! {0} {1} ",i.UniqueId,MyClass.NB);
      Int64 uid = i.UniqueId;
      //Console.WriteLine("IT IS MY TEST! {0} {1}",uid,MyClass.NB);
      max_uid = (uid>max_uid) ? uid : max_uid;
    }
    return max_uid;
  }
}
