//WARNING: this file is generated. Do not Edit
//Date 6/15/2024 10:19:21 AM
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Arcane
{
  public unsafe class ByteAbstractArray : IDisposable
  {
    [StructLayout(LayoutKind.Sequential)]
    protected unsafe struct TrueImpl
    {
      internal Integer capacity;
      internal Integer size;
      internal Byte ptr;
    }
    static protected readonly TrueImpl* EmptyVal;
    protected TrueImpl* m_p;

    static ByteAbstractArray()
    {
      EmptyVal = (TrueImpl*)Marshal.AllocHGlobal(sizeof(TrueImpl));
      EmptyVal->capacity = 0;
      EmptyVal->size = 0;
      //Console.WriteLine("STATIC CREATE ARRAY e={0}",(IntPtr)EmptyVal);
    }

    protected ByteAbstractArray()
    {
      m_p = EmptyVal;
      //Console.WriteLine("CREATE ARRAY p={0} e={1}",(IntPtr)m_p,(IntPtr)EmptyVal);
    }
 
    ~ByteAbstractArray()
    {
      //TODO: faire un test and set atomic
      //Console.WriteLine("Warning: missing call to Dispose for ByteArray");
      if (m_p!=EmptyVal){
        TrueImpl* old_v = m_p;
        //m_p = EmptyVal;
        ArrayImplBase.deallocate(sizeof(TrueImpl),sizeof(Byte),(ArrayImplBase*)old_v);
      }
    }

    public void Dispose()
    {
      if (m_p!=EmptyVal){
        TrueImpl* old_v = m_p;
        m_p = EmptyVal;
        ArrayImplBase.deallocate(sizeof(TrueImpl),sizeof(Byte),(ArrayImplBase*)old_v);
        // TODO: ajouter test interdisant d'utiliser le tableau une fois le dispose effectue
        // car on a supprime le finalize et il ne seront donc plus appelés et la mémoire
        // allouée plus jamais libérée.
        GC.SuppressFinalize(this);
      }
    }

    public Integer Size { get { return m_p->size; } }
    //! Nombre d'éléments du vecteur
    public Int32 Length { get { return (Int32)m_p->size; } }
    //! Nombre d'éléments du vecteur
    public Int64 LongLength { get { return m_p->size; } }
    //! Capacité (nombre d'éléments alloués) du vecteur
    public Integer Capacity { get { return m_p->capacity; } }
    //! Capacité (nombre d'éléments alloués) du vecteur
    public bool Empty { get { return m_p->size==0; } }
 
    protected void _Resize(Integer s)
    {
      if (s>m_p->size){
        _Realloc(s,false);
      }
      else{
      }
      m_p->size = s;
    }

    protected void _Fill(Byte v)
    {
      //FIXME utiliser un memcpy.
      Byte* p = &m_p->ptr;
      for( Integer i=0, n=m_p->size; i<n; ++i )
        p[i] = v;
    }

    protected void _Copy(Byte* rhs_begin)
    {
      //FIXME: utiliser memcpy
      Byte* p = &m_p->ptr;
      for( Integer i=0, n=m_p->size; i<n; ++i )
        p[i] = rhs_begin[i];
    }

    /*!
     * \brief Réalloue le tableau pour une nouvelle capacité égale à \a new_capacity.
     *
     * Si la nouvelle capacité est inférieure à l'ancienne, rien ne se passe.
     */
    protected void _Realloc(Integer new_capacity,bool compute_capacity)
    {
      if (m_p==EmptyVal){
        //Console.WriteLine("REALLOC EMPTY VAL wanted_new={0}",new_capacity);
        if (new_capacity!=0)
          _Allocate(new_capacity);
        return;
      }

      //Console.WriteLine("REALLOC current_capacity={0}",m_p->capacity);
      Integer capacity = new_capacity;
      if (compute_capacity){
        capacity = m_p->capacity;
        //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
        while (new_capacity>capacity)
          capacity = (capacity==0) ? 4 : (capacity + 1 + capacity / 2);
        //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
      }
      // Si la nouvelle capacitÃ© est infÃ©rieure Ã  la courante,ne fait rien.
      if (capacity<m_p->capacity)
        return;
      _Reallocate(capacity);
    }

    //! Réallocation pour un type POD
    protected void _Reallocate(Integer new_capacity)
    {
      //TrueImpl* old_p = m_p;
      m_p = (TrueImpl*)ArrayImplBase.reallocate(sizeof(TrueImpl),
                                                new_capacity,sizeof(Byte),(ArrayImplBase*)m_p);
      //m_p = (TrueImpl*)m_p->allocator->reallocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p);
      //Console.WriteLine("REALLOCATE new={0}",new_capacity);
      m_p->capacity = new_capacity;
    }

    protected void _Allocate(Integer new_capacity)
    {
      m_p = (TrueImpl*)ArrayImplBase.allocate(sizeof(TrueImpl),new_capacity,sizeof(Byte),
                                              (ArrayImplBase*)m_p);
      //m_p = (TrueImpl*)m_p->allocator->allocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p);
      //Console.WriteLine("ALLOCATE new={0}",new_capacity);
      m_p->capacity = new_capacity;
    }

    public ByteConstArrayView ConstView
    {
      get { return new ByteConstArrayView(&m_p->ptr,m_p->size); }
    }

    public ByteArrayView View
    {
      get { return new ByteArrayView(&m_p->ptr,m_p->size); }
    }
  }

  ///<summary>
  /// Tableau d'éléments de type 'Byte'
  ///</summary>
  public unsafe class ByteArray : ByteAbstractArray
  {
    /// <summary>
    /// Construit un tableau vide
    /// </summary>
    public ByteArray()
    {
    }

    /// <summary>
    /// Construit un tableau de \a n éléments non initialisés
    /// </summary>
    public ByteArray(Integer size)
    {
      _Resize(size);
    }
 
    ~ByteArray()
    {
      // La liberation de la memoire se fait dans la classe de base
      //Console.WriteLine("FINALIZE ARRAY");
    }

    /// <summary>
    /// Ajoute une valeur en fin de tableau
    /// </summary>
    /// <param name="val">
    /// A <see cref="Int32"/>
    /// </param>
    public void Add(Byte val)
    {
      //Console.WriteLine("ADD size={0} capacity={1}",m_p->size,m_p->capacity);
      if (m_p->size >= m_p->capacity)
        _Realloc(m_p->size+1,true);
      (&m_p->ptr)[m_p->size] = val;
      ++m_p->size;
    }

    /// <summary>
    /// Supprime tous les Ã©lÃ©ments du tableau
    /// </summary>
    public void Clear()
    {
      m_p->size = 0;
    }

    public void Fill(Byte v)
    {
      _Fill(v);
    }

    public void Copy(ByteConstArrayView rhs)
    {
      Byte* rhs_begin = rhs.m_ptr;
      Integer rhs_size = rhs.Size;
      Byte* begin = &m_p->ptr;
      // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
      if (begin>=rhs_begin && begin<(rhs_begin+rhs_size))
        throw new ApplicationException("Overlap error in copy");
      Resize(rhs_size);
      _Copy(rhs_begin);
    }

    //! Change le nombre d'éléments du tableau à \a s
    public void Resize(Int32 s) { _Resize(s); }
    //! Change le nombre d'éléments du tableau à \a s
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
    public Byte this[Int32 index]
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
    public Byte this[Int64 index]
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
  public unsafe struct ByteArrayView : System.Collections.Generic.IEnumerable<Byte>
  {  
    /// Enumerateur du tableau
    public struct Enumerator : IEnumerator<Byte>
    {
      internal Integer m_current;
      internal Integer m_end;
      internal Byte* m_values;

      public Enumerator(Byte* values,Integer size)
      {
        m_current = -1;
        m_end = size;
        m_values = values;
      }
      public void Reset() { m_current = -1; }
      public Byte Current { get{ return m_values[m_current]; } }
      Byte IEnumerator<Byte>.Current { get{ return m_values[m_current]; } }
      object IEnumerator.Current { get{ return m_values[m_current]; } }
      void IDisposable.Dispose(){}
      public bool MoveNext()
      {
        ++m_current;
        return m_current<m_end;
      }
    }

    internal Integer m_size;
    internal Byte* m_ptr;
    internal ByteArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = (Byte*)v.m_ptr;
    }
    internal ByteArrayView(Byte* ptr,Int32 size)
    {
      m_size = size;
      m_ptr = ptr;
    }
    internal ByteArrayView(Byte* ptr,Int64 size)
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
    IEnumerator<Byte> IEnumerable<Byte>.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    IEnumerator IEnumerable.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    [Obsolete]
    public Byte* _UnguardedBasePointer() { return m_ptr; }
    internal Byte* _InternalData() { return m_ptr; }
    public Byte[] ToArray()
    {
      Byte[] a = new Byte[m_size];
      for( Integer i=0; i<m_size; ++i )
        a[i] = m_ptr[i];
      return a;
    }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }

    public Byte this[Int32 index]
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

    public Byte this[Int64 index]
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

    //! Opérateur de conversion vers la version template
    public static implicit operator ArrayView<Byte>(ByteArrayView v)
    {
      return new ArrayView<Byte>(v.m_ptr,v.m_size);
    }

    //! Opérateur de conversion vers la version template
    public static implicit operator ConstArrayView<Byte>(ByteArrayView v)
    {
      return new ConstArrayView<Byte>(v.m_ptr,v.m_size);
    }

    //! Sous-vue de cette vue commencant a \a begin et de taille \a size
    public ByteArrayView SubView(Integer begin,Integer size)
    {
      return new ByteArrayView(m_ptr+begin,size);
    }
    
    //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
    public ByteArrayView SubViewInterval(Integer index,Integer nb_interval)
    {
      Integer n = m_size;
      Integer isize = n / nb_interval;
      Integer ibegin = index * isize;
      // Pour le dernier interval, prend les elements restants
      if ((index+1)==nb_interval)
        isize = n - ibegin;
      return new ByteArrayView(m_ptr+ibegin,isize);
    }
   
    public void Copy(ByteConstArrayView rhs)
    {
      Byte* rhs_begin = rhs.m_ptr;
      Integer rhs_size = rhs.Size;
      Byte* begin = m_ptr;
      // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
      if (begin>=rhs_begin && begin<(rhs_begin+rhs_size))
        throw new ApplicationException("Overlap error in copy");
      _Copy(rhs_begin);
    }

    internal void _Copy(Byte* rhs_begin)
    {
      //FIXME: utiliser memcpy
      Byte* p = m_ptr;
      for( Integer i=0, n=m_size; i<n; ++i )
        p[i] = rhs_begin[i];
    }
    
    public void ReadBytes(byte[] bytes,int offset)
    {
      ByteArrayView values = this;
      Integer nb_value = values.Size;
      int type_size = sizeof(Byte);
      fixed(byte* bytes_array = &bytes[offset]){
        for( Integer i=0; i<nb_value; ++i ){
          Byte v = *((Byte*)(bytes_array+i*type_size));
          values[i] = v;
        }
      }
    }

    public void Fill(Byte value)
    {
      Integer size = m_size;
      Byte* begin = m_ptr;
      for( Integer i=0; i<size; ++i )
        begin[i] = value;
    }

    public Byte At(Integer index)
    {
      return m_ptr[index];
    }
    public ByteConstArrayView ConstView { get { return new ByteConstArrayView(m_ptr,m_size); } }
	
    /// Permet de wrapper un tableau C# en une vue Arcane
    public class Wrapper : IDisposable
    {
      ByteArrayView m_view;
      GCHandle m_handle;
      public Wrapper(Byte[] array)
      {
        m_handle = GCHandle.Alloc(array,GCHandleType.Pinned);
        IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array,0);
        m_view = new ByteArrayView((Byte*)ptr,array.Length);
      }
      public static implicit operator ByteArrayView(Wrapper w)
      {
        return w.m_view;
      }
      public ByteArrayView View
      {
        get { return m_view; }
      }
      public static implicit operator ByteConstArrayView(Wrapper w)
      {
        return new ByteConstArrayView(w.m_view);
      }
      public ByteConstArrayView ConstView
      {
        get { return new ByteConstArrayView(m_view); }
      }
      void IDisposable.Dispose()
      {
        m_handle.Free();
      }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ByteConstArrayView : IEnumerable<Byte>
  {  
    /// Enumerateur du tableau
    public struct Enumerator : IEnumerator<Byte>
    {
      internal Integer m_current;
      internal Integer m_end;
      internal Byte* m_values;

      public Enumerator(Byte* values,Integer size)
      {
        m_current = -1;
        m_end = size;
        m_values = values;
      }
      public void Reset() { m_current = -1; }
      public Byte Current { get{ return m_values[m_current]; } }
      Byte IEnumerator<Byte>.Current { get{ return m_values[m_current]; } }
      object IEnumerator.Current { get{ return m_values[m_current]; } }
      void IDisposable.Dispose(){}
      public bool MoveNext()
      {
        ++m_current;
        return m_current<m_end;
      }
    }

    internal Integer m_size;
    internal Byte* m_ptr;
    public ByteConstArrayView(ByteConstArrayView v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
    internal ByteConstArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = (Byte*)v.m_ptr;
    }
    internal ByteConstArrayView(Byte* ptr,Integer size)
    {
      m_size = size;
      m_ptr = ptr;
    }
    /// Nombre d'elements du tableau
    public Integer Length { get { return m_size; } }
    /// Nombre d'elements du tableau
    public Integer Size { get { return m_size; } }
    public Enumerator GetEnumerator() { return new Enumerator(m_ptr,m_size); }
    IEnumerator<Byte> IEnumerable<Byte>.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    IEnumerator IEnumerable.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    [Obsolete]
    public Byte* _UnguardedBasePointer() { return m_ptr; }
    internal Byte* _InternalData() { return m_ptr; }
    public Byte[] ToArray()
    {
      Byte[] a = new Byte[m_size];
      for( Integer i=0; i<m_size; ++i )
        a[i] = m_ptr[i];
      return a;
    }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public ByteConstArrayView(ByteArrayView v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
    public Byte this[Integer index]
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

    //! Opérateur de conversion vers la version template
    public static implicit operator ConstArrayView<Byte>(ByteConstArrayView v)
    {
      return new ConstArrayView<Byte>(v.m_ptr,v.m_size);
    }

    //! Sous-vue de cette vue commencant a \a begin et de taille \a size
    public ByteConstArrayView SubView(Integer begin,Integer size)
    {
      return new ByteConstArrayView(m_ptr+begin,size);
    }
    
    //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
    public ByteConstArrayView SubViewInterval(Integer index,Integer nb_interval)
    {
      Integer n = m_size;
      Integer isize = n / nb_interval;
      Integer ibegin = index * isize;
      // Pour le dernier interval, prend les elements restants
      if ((index+1)==nb_interval)
        isize = n - ibegin;
      return new ByteConstArrayView(m_ptr+ibegin,isize);
    }
  }


  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ByteArray2View
  {
    internal Byte* m_ptr;
    internal Integer m_dim1_size;
    internal Integer m_dim2_size;
    public ByteArray2View(ByteArray2View v)
    {
      m_ptr = v.m_ptr;
      m_dim1_size = v.m_dim1_size;
      m_dim2_size = v.m_dim2_size;
    }
    internal ByteArray2View(Byte* ptr,Integer dim1_size,Integer dim2_size)
    {
      m_ptr = ptr;
      m_dim1_size = dim1_size;
      m_dim2_size = dim2_size;
    }
    /// Nombre d'elements de la premiere dimension du tableau
    public Integer Dim1Size { get { return m_dim1_size; } }
    /// Nombre d'elements de la deuxieme dimension du tableau
    public Integer Dim2Size { get { return m_dim2_size; } }
    [Obsolete]
    public Byte* _UnguardedBasePointer() { return m_ptr; }
    internal Byte* _InternalData() { return m_ptr; }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public ByteArrayView this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_dim1_size)
	       throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_dim1_size));
#endif
        return new ByteArrayView(m_ptr+(m_dim2_size*index),m_dim2_size);
      }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ByteConstArray2View
  {
    internal Byte* m_ptr;
    internal Integer m_dim1_size;
    internal Integer m_dim2_size;
    public ByteConstArray2View(ByteArray2View v)
    {
      m_ptr = v.m_ptr;
      m_dim1_size = v.m_dim1_size;
      m_dim2_size = v.m_dim2_size;
    }
    internal ByteConstArray2View(Byte* ptr,Integer dim1_size,Integer dim2_size)
    {
      m_ptr = ptr;
      m_dim1_size = dim1_size;
      m_dim2_size = dim2_size;
    }
    /// Nombre d'elements de la premiere dimension du tableau
    public Integer Dim1Size { get { return m_dim1_size; } }
    /// Nombre d'elements de la deuxieme dimension du tableau
    public Integer Dim2Size { get { return m_dim2_size; } }
    [Obsolete]
    public Byte* _UnguardedBasePointer() { return m_ptr; }
    internal Byte* _InternalData() { return m_ptr; }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public ByteConstArrayView this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_dim1_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_dim1_size));
#endif
        return new ByteConstArrayView(m_ptr+(m_dim2_size*index),m_dim2_size);
      }
    }
  }
}

