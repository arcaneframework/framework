using System.Runtime.InteropServices;
using System;
using Arcane;
using Integer = System.Int32;

namespace Arcane.Materials
{
  public unsafe struct ComponentItemEnumerator<T> where T : struct, IComponentItem
  {
    internal Integer m_index;
    internal Integer m_size;
    internal ComponentItemInternal** m_items;
    internal MatVarIndex* m_matvar_indexes;
    internal IntPtr m_component; // IMeshComponent* m_component;
    internal T m_true_type;
    internal ComponentItemEnumerator(ComponentItemVectorView view, T true_type)
    {
      m_items = view.m_constituent_list_view.m_items_internal.m_ptr;
      m_matvar_indexes = view.m_matvar_indexes_view.m_ptr;
      m_index = -1;
      m_size = view.m_constituent_list_view.m_items_internal.m_size;
      m_component = view.m_component;
      m_true_type = true_type;
    }
    internal ComponentItemEnumerator(IMeshComponent mesh_component, T true_type)
    : this(mesh_component.View(),true_type)
    {
    }
    public bool MoveNext()
    {
      ++m_index;
      if (m_index >= m_size)
        return false;
      m_true_type.Internal = m_items[m_index];
      return true;
    }

    public T Current
    {
      get { return m_true_type; }
    }

  }

  public static unsafe class ComponentItemEnumeratorBuilder
  {
    public static ComponentItemEnumerator<ComponentItem> Create(IMeshComponent mesh_component)
    {
      return new ComponentItemEnumerator<ComponentItem>(mesh_component, new ComponentItem(null));
    }

    public static ComponentItemEnumerator<ComponentItem> Create(IMeshComponent mesh_component, ComponentItem item)
    {
      return new ComponentItemEnumerator<ComponentItem>(mesh_component, item);
    }
    public static ComponentItemEnumerator<EnvItem> Create(IMeshEnvironment mesh_component, EnvItem item)
    {
      return new ComponentItemEnumerator<EnvItem>(mesh_component, item);
    }
    public static ComponentItemEnumerator<MatItem> Create(IMeshMaterial mesh_component, MatItem item)
    {
      return new ComponentItemEnumerator<MatItem>(mesh_component, item);
    }
  }

}
