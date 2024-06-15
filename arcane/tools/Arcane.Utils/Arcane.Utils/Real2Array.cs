//WARNING: this file is generated. Do not Edit
//Date 6/15/2024 10:19:21 AM
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Arcane
{
  public unsafe class Real2AbstractArray : IDisposable
  {
    [StructLayout(LayoutKind.Sequential)]
    protected unsafe struct TrueImpl
    {
      internal Integer capacity;
      internal Integer size;
      internal Real2 ptr;
    }
    static protected readonly TrueImpl* EmptyVal;
    protected TrueImpl* m_p;

    static Real2AbstractArray()
    {
      EmptyVal = (TrueImpl*)Marshal.AllocHGlobal(sizeof(TrueImpl));
      EmptyVal->capacity = 0;
      EmptyVal->size = 0;
      //Console.WriteLine("STATIC CREATE ARRAY e={0}",(IntPtr)EmptyVal);
    }

    protected Real2AbstractArray()
    {
      m_p = EmptyVal;
      //Console.WriteLine("CREATE ARRAY p={0} e={1}",(IntPtr)m_p,(IntPtr)EmptyVal);
    }
 
    ~Real2AbstractArray()
    {
      //TODO: faire un test and set atomic
      //Console.WriteLine("Warning: missing call to Dispose for Real2Array");
      if (m_p!=EmptyVal){
        TrueImpl* old_v = m_p;
        //m_p = EmptyVal;
        ArrayImplBase.deallocate(sizeof(TrueImpl),sizeof(Real2),(ArrayImplBase*)old_v);
      }
    }

    public void Dispose()
    {
      if (m_p!=EmptyVal){
        TrueImpl* old_v = m_p;
        m_p = EmptyVal;
        ArrayImplBase.deallocate(sizeof(TrueImpl),sizeof(Real2),(ArrayImplBase*)old_v);
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

    protected void _Fill(Real2 v)
    {
      //FIXME utiliser un memcpy.
      Real2* p = &m_p->ptr;
      for( Integer i=0, n=m_p->size; i<n; ++i )
        p[i] = v;
    }

    protected void _Copy(Real2* rhs_begin)
    {
      //FIXME: utiliser memcpy
      Real2* p = &m_p->ptr;
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
                                                new_capacity,sizeof(Real2),(ArrayImplBase*)m_p);
      //m_p = (TrueImpl*)m_p->allocator->reallocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p);
      //Console.WriteLine("REALLOCATE new={0}",new_capacity);
      m_p->capacity = new_capacity;
    }

    protected void _Allocate(Integer new_capacity)
    {
      m_p = (TrueImpl*)ArrayImplBase.allocate(sizeof(TrueImpl),new_capacity,sizeof(Real2),
                                              (ArrayImplBase*)m_p);
      //m_p = (TrueImpl*)m_p->allocator->allocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p);
      //Console.WriteLine("ALLOCATE new={0}",new_capacity);
      m_p->capacity = new_capacity;
    }

    public Real2ConstArrayView ConstView
    {
      get { return new Real2ConstArrayView(&m_p->ptr,m_p->size); }
    }

    public Real2ArrayView View
    {
      get { return new Real2ArrayView(&m_p->ptr,m_p->size); }
    }
  }

  ///<summary>
  /// Tableau d'éléments de type 'Real2'
  ///</summary>
  public unsafe class Real2Array : Real2AbstractArray
  {
    /// <summary>
    /// Construit un tableau vide
    /// </summary>
    public Real2Array()
    {
    }

    /// <summary>
    /// Construit un tableau de \a n éléments non initialisés
    /// </summary>
    public Real2Array(Integer size)
    {
      _Resize(size);
    }
 
    ~Real2Array()
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
    public void Add(Real2 val)
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

    public void Fill(Real2 v)
    {
      _Fill(v);
    }

    public void Copy(Real2ConstArrayView rhs)
    {
      Real2* rhs_begin = rhs.m_ptr;
      Integer rhs_size = rhs.Size;
      Real2* begin = &m_p->ptr;
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
    public Real2 this[Int32 index]
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
    public Real2 this[Int64 index]
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
  public unsafe struct Real2ArrayView : System.Collections.Generic.IEnumerable<Real2>
  {  
    /// Enumerateur du tableau
    public struct Enumerator : IEnumerator<Real2>
    {
      internal Integer m_current;
      internal Integer m_end;
      internal Real2* m_values;

      public Enumerator(Real2* values,Integer size)
      {
        m_current = -1;
        m_end = size;
        m_values = values;
      }
      public void Reset() { m_current = -1; }
      public Real2 Current { get{ return m_values[m_current]; } }
      Real2 IEnumerator<Real2>.Current { get{ return m_values[m_current]; } }
      object IEnumerator.Current { get{ return m_values[m_current]; } }
      void IDisposable.Dispose(){}
      public bool MoveNext()
      {
        ++m_current;
        return m_current<m_end;
      }
    }

    internal Integer m_size;
    internal Real2* m_ptr;
    internal Real2ArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = (Real2*)v.m_ptr;
    }
    internal Real2ArrayView(Real2* ptr,Int32 size)
    {
      m_size = size;
      m_ptr = ptr;
    }
    internal Real2ArrayView(Real2* ptr,Int64 size)
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
    IEnumerator<Real2> IEnumerable<Real2>.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    IEnumerator IEnumerable.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    [Obsolete]
    public Real2* _UnguardedBasePointer() { return m_ptr; }
    internal Real2* _InternalData() { return m_ptr; }
    public Real2[] ToArray()
    {
      Real2[] a = new Real2[m_size];
      for( Integer i=0; i<m_size; ++i )
        a[i] = m_ptr[i];
      return a;
    }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }

    public Real2 this[Int32 index]
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

    public Real2 this[Int64 index]
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
    public static implicit operator ArrayView<Real2>(Real2ArrayView v)
    {
      return new ArrayView<Real2>(v.m_ptr,v.m_size);
    }

    //! Opérateur de conversion vers la version template
    public static implicit operator ConstArrayView<Real2>(Real2ArrayView v)
    {
      return new ConstArrayView<Real2>(v.m_ptr,v.m_size);
    }

    //! Sous-vue de cette vue commencant a \a begin et de taille \a size
    public Real2ArrayView SubView(Integer begin,Integer size)
    {
      return new Real2ArrayView(m_ptr+begin,size);
    }
    
    //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
    public Real2ArrayView SubViewInterval(Integer index,Integer nb_interval)
    {
      Integer n = m_size;
      Integer isize = n / nb_interval;
      Integer ibegin = index * isize;
      // Pour le dernier interval, prend les elements restants
      if ((index+1)==nb_interval)
        isize = n - ibegin;
      return new Real2ArrayView(m_ptr+ibegin,isize);
    }
   
    public void Copy(Real2ConstArrayView rhs)
    {
      Real2* rhs_begin = rhs.m_ptr;
      Integer rhs_size = rhs.Size;
      Real2* begin = m_ptr;
      // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
      if (begin>=rhs_begin && begin<(rhs_begin+rhs_size))
        throw new ApplicationException("Overlap error in copy");
      _Copy(rhs_begin);
    }

    internal void _Copy(Real2* rhs_begin)
    {
      //FIXME: utiliser memcpy
      Real2* p = m_ptr;
      for( Integer i=0, n=m_size; i<n; ++i )
        p[i] = rhs_begin[i];
    }
    
    public void ReadBytes(byte[] bytes,int offset)
    {
      Real2ArrayView values = this;
      Integer nb_value = values.Size;
      int type_size = sizeof(Real2);
      fixed(byte* bytes_array = &bytes[offset]){
        for( Integer i=0; i<nb_value; ++i ){
          Real2 v = *((Real2*)(bytes_array+i*type_size));
          values[i] = v;
        }
      }
    }

    public void Fill(Real2 value)
    {
      Integer size = m_size;
      Real2* begin = m_ptr;
      for( Integer i=0; i<size; ++i )
        begin[i] = value;
    }

    public Real2 At(Integer index)
    {
      return m_ptr[index];
    }
    public Real2ConstArrayView ConstView { get { return new Real2ConstArrayView(m_ptr,m_size); } }
	
    /// Permet de wrapper un tableau C# en une vue Arcane
    public class Wrapper : IDisposable
    {
      Real2ArrayView m_view;
      GCHandle m_handle;
      public Wrapper(Real2[] array)
      {
        m_handle = GCHandle.Alloc(array,GCHandleType.Pinned);
        IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array,0);
        m_view = new Real2ArrayView((Real2*)ptr,array.Length);
      }
      public static implicit operator Real2ArrayView(Wrapper w)
      {
        return w.m_view;
      }
      public Real2ArrayView View
      {
        get { return m_view; }
      }
      public static implicit operator Real2ConstArrayView(Wrapper w)
      {
        return new Real2ConstArrayView(w.m_view);
      }
      public Real2ConstArrayView ConstView
      {
        get { return new Real2ConstArrayView(m_view); }
      }
      void IDisposable.Dispose()
      {
        m_handle.Free();
      }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Real2ConstArrayView : IEnumerable<Real2>
  {  
    /// Enumerateur du tableau
    public struct Enumerator : IEnumerator<Real2>
    {
      internal Integer m_current;
      internal Integer m_end;
      internal Real2* m_values;

      public Enumerator(Real2* values,Integer size)
      {
        m_current = -1;
        m_end = size;
        m_values = values;
      }
      public void Reset() { m_current = -1; }
      public Real2 Current { get{ return m_values[m_current]; } }
      Real2 IEnumerator<Real2>.Current { get{ return m_values[m_current]; } }
      object IEnumerator.Current { get{ return m_values[m_current]; } }
      void IDisposable.Dispose(){}
      public bool MoveNext()
      {
        ++m_current;
        return m_current<m_end;
      }
    }

    internal Integer m_size;
    internal Real2* m_ptr;
    public Real2ConstArrayView(Real2ConstArrayView v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
    internal Real2ConstArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = (Real2*)v.m_ptr;
    }
    internal Real2ConstArrayView(Real2* ptr,Integer size)
    {
      m_size = size;
      m_ptr = ptr;
    }
    /// Nombre d'elements du tableau
    public Integer Length { get { return m_size; } }
    /// Nombre d'elements du tableau
    public Integer Size { get { return m_size; } }
    public Enumerator GetEnumerator() { return new Enumerator(m_ptr,m_size); }
    IEnumerator<Real2> IEnumerable<Real2>.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    IEnumerator IEnumerable.GetEnumerator()
    { return new Enumerator(m_ptr,m_size); }
    [Obsolete]
    public Real2* _UnguardedBasePointer() { return m_ptr; }
    internal Real2* _InternalData() { return m_ptr; }
    public Real2[] ToArray()
    {
      Real2[] a = new Real2[m_size];
      for( Integer i=0; i<m_size; ++i )
        a[i] = m_ptr[i];
      return a;
    }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public Real2ConstArrayView(Real2ArrayView v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
    public Real2 this[Integer index]
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
    public static implicit operator ConstArrayView<Real2>(Real2ConstArrayView v)
    {
      return new ConstArrayView<Real2>(v.m_ptr,v.m_size);
    }

    //! Sous-vue de cette vue commencant a \a begin et de taille \a size
    public Real2ConstArrayView SubView(Integer begin,Integer size)
    {
      return new Real2ConstArrayView(m_ptr+begin,size);
    }
    
    //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
    public Real2ConstArrayView SubViewInterval(Integer index,Integer nb_interval)
    {
      Integer n = m_size;
      Integer isize = n / nb_interval;
      Integer ibegin = index * isize;
      // Pour le dernier interval, prend les elements restants
      if ((index+1)==nb_interval)
        isize = n - ibegin;
      return new Real2ConstArrayView(m_ptr+ibegin,isize);
    }
  }


  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Real2Array2View
  {
    internal Real2* m_ptr;
    internal Integer m_dim1_size;
    internal Integer m_dim2_size;
    public Real2Array2View(Real2Array2View v)
    {
      m_ptr = v.m_ptr;
      m_dim1_size = v.m_dim1_size;
      m_dim2_size = v.m_dim2_size;
    }
    internal Real2Array2View(Real2* ptr,Integer dim1_size,Integer dim2_size)
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
    public Real2* _UnguardedBasePointer() { return m_ptr; }
    internal Real2* _InternalData() { return m_ptr; }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public Real2ArrayView this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_dim1_size)
	       throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_dim1_size));
#endif
        return new Real2ArrayView(m_ptr+(m_dim2_size*index),m_dim2_size);
      }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Real2ConstArray2View
  {
    internal Real2* m_ptr;
    internal Integer m_dim1_size;
    internal Integer m_dim2_size;
    public Real2ConstArray2View(Real2Array2View v)
    {
      m_ptr = v.m_ptr;
      m_dim1_size = v.m_dim1_size;
      m_dim2_size = v.m_dim2_size;
    }
    internal Real2ConstArray2View(Real2* ptr,Integer dim1_size,Integer dim2_size)
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
    public Real2* _UnguardedBasePointer() { return m_ptr; }
    internal Real2* _InternalData() { return m_ptr; }
    [Obsolete]
    public IntPtr _Base()
    {
      return (IntPtr)m_ptr;
    }
    public Real2ConstArrayView this[Integer index]
    {
      get
      {
#if !ARCANE_UNSAFE
        if (index<0 || index>=m_dim1_size)
          throw new ArgumentOutOfRangeException(String.Format("bad index i={0} max={1}",index,m_dim1_size));
#endif
        return new Real2ConstArrayView(m_ptr+(m_dim2_size*index),m_dim2_size);
      }
    }
  }
}

