using System.Runtime.InteropServices;
using System;

namespace Arcane.Materials
{
  static class ListSizeChecker
  {
    internal static void Check(int index, int size)
    {
      if (index < 0 || index >= size)
        throw new ArgumentOutOfRangeException("index", index, "Bad component index");
    }
  }
  interface IMeshComponentCommon<T>
  {
    int Size { get; }
    T GetItem(int index);

  }
  [StructLayout(LayoutKind.Sequential)]
  internal unsafe struct IMeshEnvironmentPtr
  {
    internal IntPtr m_data;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal unsafe struct IMeshMaterialPtr
  {
    internal IntPtr m_data;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal unsafe struct IMeshComponentPtr
  {
    internal IntPtr m_data;
  }
}

namespace Arcane.Materials
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MeshEnvironmentListView : IMeshComponentCommon<IMeshEnvironment>
  {
    internal Int32 m_size;
    internal Arcane.Materials.IMeshEnvironmentPtr* m_ptr;

    public int Size { get { return m_size; } }
    public IMeshEnvironment this[int index]
    {
      get
      {
        return GetItem(index);
      }
    }
    public IMeshEnvironment GetItem(int index)
    {
      ListSizeChecker.Check(index, m_size);
      return new IMeshEnvironment(m_ptr[index].m_data, false);
    }
    public MeshComponentEnumerator<IMeshEnvironment> GetEnumerator()
    {
      return new MeshComponentEnumerator<IMeshEnvironment>(this);
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MeshMaterialListView : IMeshComponentCommon<IMeshMaterial>
  {
    internal Int32 m_size;
    internal Arcane.Materials.IMeshMaterialPtr* m_ptr;

    public int Size { get { return m_size; } }
    public Arcane.Materials.IMeshMaterial this[int index]
    {
      get { return GetItem(index); }
    }
    public IMeshMaterial GetItem(int index)
    {
      ListSizeChecker.Check(index, m_size);
      return new IMeshMaterial(m_ptr[index].m_data, false);
    }
    public MeshComponentEnumerator<IMeshMaterial> GetEnumerator()
    {
      return new MeshComponentEnumerator<IMeshMaterial>(this);
    }

  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MeshComponentListView : IMeshComponentCommon<IMeshComponent>
  {
    internal Int32 m_size;
    internal Arcane.Materials.IMeshComponentPtr* m_ptr;

    public int Size { get { return m_size; } }
    public IMeshComponent this[int index]
    {
      get { return GetItem(index); }
    }
    public IMeshComponent GetItem(int index)
    {
      ListSizeChecker.Check(index, m_size);
      return new IMeshComponent(m_ptr[index].m_data, false);
    }

    public MeshComponentEnumerator<IMeshComponent> GetEnumerator()
    {
      return new MeshComponentEnumerator<IMeshComponent>(this);
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MeshComponentEnumerator<T>
  {
    internal int m_index;
    internal int m_size;
    internal IMeshComponentCommon<T> m_view;

    internal MeshComponentEnumerator(IMeshComponentCommon<T> view)
    {
      m_view = view;
      m_index = -1;
      m_size = view.Size;
    }

    public bool MoveNext()
    {
      ++m_index;
      return m_index < m_size;
    }

    public T Current
    {
      get { return m_view.GetItem(m_index); }
    }
  }
}