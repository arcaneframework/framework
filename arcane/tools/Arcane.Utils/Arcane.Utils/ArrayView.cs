//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Arcane
{
  public unsafe class AbstractArray<CTYPE> : IDisposable where CTYPE : unmanaged
  {
    [StructLayout(LayoutKind.Sequential)]
    protected unsafe struct TrueImpl
    {
      internal Int64 capacity;
      internal Int64 size;
      internal CTYPE ptr;
    }
    static protected readonly TrueImpl* EmptyVal;
    protected TrueImpl* m_p;

    static AbstractArray()
    {
      EmptyVal = (TrueImpl*)Marshal.AllocHGlobal(sizeof(TrueImpl));
      EmptyVal->capacity = 0;
      EmptyVal->size = 0;
      //Console.WriteLine("STATIC CREATE ARRAY e={0}",(IntPtr)EmptyVal);
    }

    protected AbstractArray()
    {
      m_p = EmptyVal;
      //Console.WriteLine("CREATE ARRAY p={0} e={1}",(IntPtr)m_p,(IntPtr)EmptyVal);
    }
 
    ~AbstractArray()
    {
      //TODO: faire un test and set atomic
      //Console.WriteLine("Warning: missing call to Dispose for @CTYPE@Array");
      if (m_p!=EmptyVal){
        TrueImpl* old_v = m_p;
        //m_p = EmptyVal;
        ArrayImplBase.deallocate(sizeof(TrueImpl),sizeof(CTYPE),(ArrayImplBase*)old_v);
      }
    }

    public void Dispose()
    {
      if (m_p!=EmptyVal){
        TrueImpl* old_v = m_p;
        m_p = EmptyVal;
        ArrayImplBase.deallocate(sizeof(TrueImpl),sizeof(CTYPE),(ArrayImplBase*)old_v);
        // TODO: ajouter test interdisant d'utiliser le tableau une fois le dispose effectué
        // car on a supprime le finalize et il ne sera donc plus appelé et la mémoire
        // allouée plus jamais liberée.
        GC.SuppressFinalize(this);
      }
    }

    //! Nombre d'éléments du vecteur (sur 32 bits)
    public Int32 Size { get { return (Int32)m_p->size; } }
    //! Nombre d'éléments du vecteur
    public Int64 LargeSize { get { return m_p->size; } }
    //! Nombre d'éléments du vecteur
    public Int32 Length { get { return (Int32)m_p->size; } }
    //! Nombre d'éléments du vecteur
    public Int64 LongLength { get { return m_p->size; } }
    //! Capacité (nombre d'éléments alloués) du vecteur
    public Integer Capacity { get { return (Int32)m_p->capacity; } }
    //! Capacité (nombre d'éléments alloués) du vecteur
    public Int64 LargeCapacity { get { return m_p->capacity; } }
    //! Capacité (nombre d'éléments alloués) du vecteur
    public bool Empty { get { return m_p->size==0; } }
 
    protected void _Resize(Int64 s)
    {
      if (s>m_p->size){
        _Realloc(s,false);
      }
      m_p->size = s;
    }

    protected void _Fill(CTYPE v)
    {
      //FIXME utiliser un memcpy.
      CTYPE* p = &m_p->ptr;
      for( Int64 i=0, n=m_p->size; i<n; ++i )
        p[i] = v;
    }

    protected void _Copy(CTYPE* rhs_begin)
    {
      //FIXME: utiliser memcpy
      CTYPE* p = &m_p->ptr;
      for( Int64 i=0, n=m_p->size; i<n; ++i )
        p[i] = rhs_begin[i];
    }

    /*!
     * \brief Réalloue le tableau pour une nouvelle capacité égale à `new_capacity`.
     *
     * Si la nouvelle capacité est inférieure à l'ancienne, rien ne se passe.
     */
    protected void _Realloc(Int64 new_capacity,bool compute_capacity)
    {
      if (m_p==EmptyVal){
        //Console.WriteLine("REALLOC EMPTY VAL wanted_new={0}",new_capacity);
        if (new_capacity!=0)
          _Allocate(new_capacity);
        return;
      }

      //Console.WriteLine("REALLOC current_capacity={0}",m_p->capacity);
      Int64 capacity = new_capacity;
      if (compute_capacity){
        capacity = m_p->capacity;
        //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
        while (new_capacity>capacity)
          capacity = (capacity==0) ? 4 : (capacity + 1 + capacity / 2);
        //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
      }
      // Si la nouvelle capacité est inférieure à la courante, ne fait rien.
      if (capacity<m_p->capacity)
        return;
      _Reallocate(capacity);
    }

    //! Réallocation pour un type POD
    protected void _Reallocate(Int64 new_capacity)
    {
      //TrueImpl* old_p = m_p;
      m_p = (TrueImpl*)ArrayImplBase.reallocate(sizeof(TrueImpl),
                                                new_capacity,sizeof(CTYPE),(ArrayImplBase*)m_p);
      //m_p = (TrueImpl*)m_p->allocator->reallocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p);
      //Console.WriteLine("REALLOCATE new={0}",new_capacity);
      m_p->capacity = new_capacity;
    }

    protected void _Allocate(Int64 new_capacity)
    {
      m_p = (TrueImpl*)ArrayImplBase.allocate(sizeof(TrueImpl),new_capacity,sizeof(CTYPE),
                                              (ArrayImplBase*)m_p);
      //m_p = (TrueImpl*)m_p->allocator->allocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p);
      //Console.WriteLine("ALLOCATE new={0}",new_capacity);
      m_p->capacity = new_capacity;
    }

    public ConstArrayView<CTYPE> ConstView
    {
      get { return new ConstArrayView<CTYPE>(&m_p->ptr,Size); }
    }

    public ArrayView<CTYPE> View
    {
      get { return new ArrayView<CTYPE>(&m_p->ptr,Size); }
    }
  }

  ///<summary>
  /// Tableau d'éléments de type 'CTYPE'
  ///</summary>
  public unsafe class Array<CTYPE> : AbstractArray<CTYPE> where CTYPE : unmanaged
  {
    /// <summary>
    /// Construit un tableau vide
    /// </summary>
    public Array()
    {
    }

    /// <summary>
    /// Construit un tableau de \a n éléents non initialisés
    /// </summary>
    public Array(Integer size)
    {
      _Resize(size);
    }
 
    ~Array()
    {
      // La libération de la mémoire se fait dans la classe de base
      //Console.WriteLine("FINALIZE ARRAY");
    }
    /// <summary>
    /// Ajoute une valeur en fin de tableau
    /// </summary>
    /// <param name="val">
    /// A <see cref="Int32"/>
    /// </param>
    public void Add(CTYPE val)
    {
      //Console.WriteLine("ADD size={0} capacity={1}",m_p->size,m_p->capacity);
      if (m_p->size >= m_p->capacity)
        _Realloc(m_p->size+1,true);
      (&m_p->ptr)[m_p->size] = val;
      ++m_p->size;
    }

    /// <summary>
    /// Supprime tous les éléments du tableau
    /// </summary>
    public void Clear()
    {
      m_p->size = 0;
    }

    public void Fill(CTYPE v)
    {
      _Fill(v);
    }

    public void Copy(ConstArrayView<CTYPE> rhs)
    {
      CTYPE* rhs_begin = rhs.m_ptr;
      Integer rhs_size = rhs.Size;
      CTYPE* begin = &m_p->ptr;
      // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
      if (begin>=rhs_begin && begin<(rhs_begin+rhs_size))
        throw new ApplicationException("Overlap error in copy");
      Resize(rhs_size);
      _Copy(rhs_begin);
    }

    //! Change le nombre d'élément du tableau à \a s
    public void Resize(Int32 s) { _Resize(s); }
    //! Change le nombre d'élément du tableau à \a s
    public void Resize(Int64 s) { _Resize((Integer)s); }
    //! Réserve la mémoire pour \a new_capacity éléments
    public void Reserve(Integer new_capacity)
    {
      if (new_capacity<=m_p->capacity)
        return;
      _Realloc(new_capacity,false);
    }

    /// <summary>
    /// Positionne ou récupère le index-ème élément du tableau
    /// </summary>
    public CTYPE this[Int32 index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=(Int32)m_p->size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_p->size));
#endif
        return (&m_p->ptr)[index];
      }
      set
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=(Int64)m_p->size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_p->size));
#endif
        (&m_p->ptr)[index] = value;
      }
    }
    
    /// <summary>
    /// Positionne ou récupère le index-ème élément du tableau
    /// </summary>
    public CTYPE this[Int64 index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=(Int64)m_p->size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_p->size));
#endif
        return (&m_p->ptr)[index];
      }
      set
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=(Int64)m_p->size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_p->size));
#endif
        (&m_p->ptr)[index] = value;
      }
    }
  }



  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ArrayView<CTYPE> : System.Collections.Generic.IEnumerable<CTYPE> where CTYPE : unmanaged
  {  
    /// Enumerateur du tableau
    public struct Enumerator : IEnumerator<CTYPE>
    {
      internal Integer m_current;
      internal Integer m_end;
      internal CTYPE* m_values;

      public Enumerator(CTYPE* values,Integer size)
      {
        m_current = -1;
        m_end = size;
        m_values = values;
      }
      public void Reset() { m_current = -1; }
      public CTYPE Current { get{ return m_values[m_current]; } }
      CTYPE IEnumerator<CTYPE>.Current { get{ return m_values[m_current]; } }
      object IEnumerator.Current { get{ return m_values[m_current]; } }
      void IDisposable.Dispose(){}
      public bool MoveNext()
      {
        ++m_current;
        return m_current<m_end;
      }
    }

    internal Integer m_size;
    internal CTYPE* m_ptr;
    internal ArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = (CTYPE*)v.m_ptr;
    }
    internal ArrayView(CTYPE* ptr,Int32 size)
    {
      m_size = size;
      m_ptr = ptr;
    }
    internal ArrayView(CTYPE* ptr,Int64 size)
    {
      m_size = (Integer)size;
      m_ptr = ptr;
    }
    /// Nombre d'elements du tableau
    public Int32 Length { get { return (Int32)m_size; } }
    /// Nombre d'elements du tableau
    public Int64 LongLength { get { return m_size; } }
    /// Nombre d'elements du tableau
    public Integer Size { get { return m_size; } }
    public Enumerator GetEnumerator() { return new Enumerator(m_ptr,m_size); }
    IEnumerator<CTYPE> IEnumerable<CTYPE>.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    IEnumerator IEnumerable.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    [Obsolete]
    public CTYPE* _UnguardedBasePointer() { return m_ptr; }
    public CTYPE[] ToArray()
    {
      CTYPE[] a = new CTYPE[m_size];
      for( Integer i=0; i<m_size; ++i )
        a[i] = m_ptr[i];
      return a;
    }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }

    public CTYPE this[Int32 index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_size));
#endif
        return m_ptr[index];
      }
      set
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_size));
#endif
        m_ptr[index] = value;
      }
    }

    public CTYPE this[Int64 index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_size));
#endif
        return m_ptr[index];
      }
      set
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_size));
#endif
        m_ptr[index] = value;
      }
    }

    //! Sous-vue de cette vue commencant a \a begin et de taille \a size
    public ArrayView<CTYPE> SubView(Integer begin,Integer size)
    {
      return new ArrayView<CTYPE>(m_ptr+begin,size);
    }
    
    //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
    public ArrayView<CTYPE> SubViewInterval(Integer index,Integer nb_interval)
    {
      Integer n = m_size;
      Integer isize = n / nb_interval;
      Integer ibegin = index * isize;
      // Pour le dernier interval, prend les elements restants
      if ((index+1)==nb_interval)
        isize = n - ibegin;
      return new ArrayView<CTYPE>(m_ptr+ibegin,isize);
    }
   
    public void Copy(ConstArrayView<CTYPE> rhs)
    {
      CTYPE* rhs_begin = rhs.m_ptr;
      Integer rhs_size = rhs.Size;
      CTYPE* begin = m_ptr;
      // VÃ©rifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
      if (begin>=rhs_begin && begin<(rhs_begin+rhs_size))
        throw new ApplicationException("Overlap error in copy");
      _Copy(rhs_begin);
    }

    internal void _Copy(CTYPE* rhs_begin)
    {
      //FIXME: utiliser memcpy
      CTYPE* p = m_ptr;
      for( Integer i=0, n=m_size; i<n; ++i )
        p[i] = rhs_begin[i];
    }
    
    public void ReadBytes(byte[] bytes,int offset)
    {
      ArrayView<CTYPE> values = this;
      Integer nb_value = values.Size;
      int type_size = sizeof(CTYPE);
      fixed(byte* bytes_array = &bytes[offset]){
        for( Integer i=0; i<nb_value; ++i ){
          CTYPE v = *((CTYPE*)(bytes_array+i*type_size));
          values[i] = v;
        }
      }
    }

    public void Fill(CTYPE value)
    {
      Integer size = m_size;
      CTYPE* begin = m_ptr;
      for( Integer i=0; i<size; ++i )
        begin[i] = value;
    }

    public CTYPE At(Integer index)
    {
      return m_ptr[index];
    }
    public ConstArrayView<CTYPE> ConstView { get { return new ConstArrayView<CTYPE>(m_ptr,m_size); } }
	
    /// Permet de wrapper un tableau C# en une vue Arcane
    public class Wrapper : IDisposable
    {
      ArrayView<CTYPE> m_view;
      GCHandle m_handle;
      public Wrapper(CTYPE[] array)
      {
        m_handle = GCHandle.Alloc(array,GCHandleType.Pinned);
        IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array,0);
        m_view = new ArrayView<CTYPE>((CTYPE*)ptr,array.Length);
      }
      public static implicit operator ArrayView<CTYPE>(Wrapper w)
      {
        return w.m_view;
      }
      public ArrayView<CTYPE> View
      {
        get { return m_view; }
      }
      public static implicit operator ConstArrayView<CTYPE>(Wrapper w)
      {
        return new ConstArrayView<CTYPE>(w.m_view);
      }
      public ConstArrayView<CTYPE> ConstView
      {
        get { return new ConstArrayView<CTYPE>(m_view); }
      }
      void IDisposable.Dispose()
      {
        m_handle.Free();
      }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ConstArrayView<CTYPE> : IEnumerable<CTYPE> where CTYPE : unmanaged
  {  
    /// Enumérateur du tableau
    public struct Enumerator : IEnumerator<CTYPE>
    {
      internal Integer m_current;
      internal Integer m_end;
      internal CTYPE* m_values;

      public Enumerator(CTYPE* values,Integer size)
      {
        m_current = -1;
        m_end = size;
        m_values = values;
      }
      public void Reset() { m_current = -1; }
      public CTYPE Current { get{ return m_values[m_current]; } }
      CTYPE IEnumerator<CTYPE>.Current { get{ return m_values[m_current]; } }
      object IEnumerator.Current { get{ return m_values[m_current]; } }
      void IDisposable.Dispose(){}
      public bool MoveNext()
      {
        ++m_current;
        return m_current<m_end;
      }
    }

    internal Integer m_size;
    internal CTYPE* m_ptr;
    public ConstArrayView(ConstArrayView<CTYPE> v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
    internal ConstArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = (CTYPE*)v.m_ptr;
    }
    internal ConstArrayView(CTYPE* ptr,Integer size)
    {
      m_size = size;
      m_ptr = ptr;
    }
    /// Nombre d'éléments du tableau
    public Integer Length { get { return m_size; } }
    /// Nombre d'éléments du tableau
    public Integer Size { get { return m_size; } }
    public Enumerator GetEnumerator() { return new Enumerator(m_ptr,m_size); }
    IEnumerator<CTYPE> IEnumerable<CTYPE>.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    IEnumerator IEnumerable.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    [Obsolete]
    public CTYPE* _UnguardedBasePointer() { return m_ptr; }
    public CTYPE[] ToArray()
    {
      CTYPE[] a = new CTYPE[m_size];
      for( Integer i=0; i<m_size; ++i )
        a[i] = m_ptr[i];
      return a;
    }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public ConstArrayView(ArrayView<CTYPE> v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
    public CTYPE this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_size));
#endif
        return m_ptr[index];
      }
    }
    //! Sous-vue de cette vue commencant a \a begin et de taille \a size
    public ConstArrayView<CTYPE> SubView(Integer begin,Integer size)
    {
      return new ConstArrayView<CTYPE>(m_ptr+begin,size);
    }
    
    //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
    public ConstArrayView<CTYPE> SubViewInterval(Integer index,Integer nb_interval)
    {
      Integer n = m_size;
      Integer isize = n / nb_interval;
      Integer ibegin = index * isize;
      // Pour le dernier interval, prend les elements restants
      if ((index+1)==nb_interval)
        isize = n - ibegin;
      return new ConstArrayView<CTYPE>(m_ptr+ibegin,isize);
    }
  }


  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Array2View<CTYPE> where CTYPE : unmanaged
  {
    internal CTYPE* m_ptr;
    internal Integer m_dim1_size;
    internal Integer m_dim2_size;
    public Array2View(Array2View<CTYPE> v)
    {
      m_ptr = v.m_ptr;
      m_dim1_size = v.m_dim1_size;
      m_dim2_size = v.m_dim2_size;
    }
    internal Array2View(CTYPE* ptr,Integer dim1_size,Integer dim2_size)
    {
      m_ptr = ptr;
      m_dim1_size = dim1_size;
      m_dim2_size = dim2_size;
    }
    /// Nombre d'éléments de la première dimension du tableau
    public Integer Dim1Size { get { return m_dim1_size; } }
    /// Nombre d'éléments de la deuxième dimension du tableau
    public Integer Dim2Size { get { return m_dim2_size; } }
    [Obsolete]
    public CTYPE* _UnguardedBasePointer() { return m_ptr; }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public ArrayView<CTYPE> this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_dim1_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_dim1_size));
#endif
        return new ArrayView<CTYPE>(m_ptr+(m_dim2_size*index),m_dim2_size);
      }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ConstArray2View<CTYPE> where CTYPE : unmanaged
  {
    internal CTYPE* m_ptr;
    internal Integer m_dim1_size;
    internal Integer m_dim2_size;
    public ConstArray2View(Array2View<CTYPE> v)
    {
      m_ptr = v.m_ptr;
      m_dim1_size = v.m_dim1_size;
      m_dim2_size = v.m_dim2_size;
    }
    internal ConstArray2View(CTYPE* ptr,Integer dim1_size,Integer dim2_size)
    {
      m_ptr = ptr;
      m_dim1_size = dim1_size;
      m_dim2_size = dim2_size;
    }
    /// Nombre d'éléments de la première dimension du tableau
    public Integer Dim1Size { get { return m_dim1_size; } }
    /// Nombre d'éléments de la deuxième dimension du tableau
    public Integer Dim2Size { get { return m_dim2_size; } }
    [Obsolete]
    public CTYPE* _UnguardedBasePointer() { return m_ptr; }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public ConstArrayView<CTYPE> this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_dim1_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_dim1_size));
#endif
        return new ConstArrayView<CTYPE>(m_ptr+(m_dim2_size*index),m_dim2_size);
      }
    }
  }
}
