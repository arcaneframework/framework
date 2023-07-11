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
    internal ComponentItemEnumerator(ComponentItemInternal** items, MatVarIndex* matvar_indexes, Integer size, T true_type)
    {
      m_items = items;
      m_matvar_indexes = matvar_indexes;
      m_index = -1;
      m_size = size;
      m_component = IntPtr.Zero; // TODO: mettre la bonne valeur
      m_true_type = true_type;
    }
    internal ComponentItemEnumerator(ComponentItemInternalPtrConstArrayView items, MatVarIndexConstArrayView matvar_indexes, T true_type)
    : this(items.m_ptr,matvar_indexes.m_ptr,items.m_size,true_type)
    {
    }

    internal ComponentItemEnumerator(IMeshComponent mesh_component, T true_type)
      : this(mesh_component._internalApi().ItemsInternalView(),mesh_component._internalApi().VariableIndexer().MatvarIndexes(),true_type)
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
      var indexes = mesh_component._internalApi().VariableIndexer().MatvarIndexes();
      var items = mesh_component._internalApi().ItemsInternalView();
      return new ComponentItemEnumerator<ComponentItem>(items.m_ptr, indexes.m_ptr, items.m_size, new ComponentItem(null));
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
