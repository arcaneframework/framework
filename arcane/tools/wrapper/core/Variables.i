// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapper pour les variables.
//
// Les classes Variable* de Arcane sont mappees sous le nom Variable*_INTERNAL
// et les classes C# sont generees directement par ce fichier.
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%include arcane/core/VariableBuildInfo.h
%include arcane/core/VariableRef.h

%include arcane/core/PrivateVariableScalar.h
%include arcane/core/PrivateVariableArray.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%template(VariableCommonScalarReal_INTERNAL) Arcane::PrivateVariableScalarT<Real>;
%template(VariableCommonScalarInt32_INTERNAL) Arcane::PrivateVariableScalarT<Int32>;
%template(VariableCommonScalarInt64_INTERNAL) Arcane::PrivateVariableScalarT<Int64>;
%template(VariableCommonScalarReal3_INTERNAL) Arcane::PrivateVariableScalarT<Real3>;
%template(VariableCommonScalarReal3x3_INTERNAL) Arcane::PrivateVariableScalarT<Real3x3>;
%template(VariableCommonScalarReal2_INTERNAL) Arcane::PrivateVariableScalarT<Real2>;
%template(VariableCommonScalarReal2x2_INTERNAL) Arcane::PrivateVariableScalarT<Real2x2>;

%template(VariableCommonArrayReal_INTERNAL) Arcane::PrivateVariableArrayT<Real>;
%template(VariableCommonArrayInt32_INTERNAL) Arcane::PrivateVariableArrayT<Int32>;
%template(VariableCommonArrayInt64_INTERNAL) Arcane::PrivateVariableArrayT<Int64>;
%template(VariableCommonArrayReal3_INTERNAL) Arcane::PrivateVariableArrayT<Real3>;
%template(VariableCommonArrayReal3x3_INTERNAL) Arcane::PrivateVariableArrayT<Real3x3>;
%template(VariableCommonArrayReal2_INTERNAL) Arcane::PrivateVariableArrayT<Real2>;
%template(VariableCommonArrayReal2x2_INTERNAL) Arcane::PrivateVariableArrayT<Real2x2>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(cscode) Arcane::MeshVariableRef
%{
  GCHandle saved_del_handle;
  VariableUpdateNotifier m_update_notifier;
  bool is_disposed;
  internal delegate void UpdateDelegate();

  protected void _InitMeshVariableBase()
  {
    VariableRef var = this;

    m_update_notifier = new VariableUpdateNotifier(this);
    m_update_notifier.Register(VariableRef.getCPtr(var));
    saved_del_handle = GCHandle.Alloc(m_update_notifier);

    //System.Diagnostics.StackTrace st = new System.Diagnostics.StackTrace();
    //Console.WriteLine("** C#:SetDelegate ={0}",st);
  }
  
  internal void _DirectOnSizeChanged()
  {
    if (is_disposed)
      return;
    //Console.WriteLine("** C#:Calling Size Changed for '{0}'",Name);
    //System.Diagnostics.StackTrace st = new System.Diagnostics.StackTrace();
    //Console.WriteLine("** C#:Trace ={0}",st);
    //Console.Out.Flush();
    _OnSizeChanged();
  }

  protected virtual void _OnSizeChanged()
  {
    Console.WriteLine("** WARNING: size changed for '{0}'",Name());
  }

  [DllImport("$dllimport")]
  static extern IntPtr _ArcaneWrapperCoreAddVariableChangedDelegate(HandleRef p,UpdateDelegate d);
  static internal IntPtr _AddChangedDelegate(HandleRef p,UpdateDelegate d)
  {
    return _ArcaneWrapperCoreAddVariableChangedDelegate(p,d);
  }

  [DllImport("$dllimport")]
  static extern void _ArcaneWrapperCoreRemoveVariableChangedDelegate(IntPtr p,UpdateDelegate d);
  static internal void _RemoveChangedDelegate(IntPtr p,UpdateDelegate d)
  {
    _ArcaneWrapperCoreRemoveVariableChangedDelegate(p,d);
  }
%}

%{
  extern "C" ARCANE_CORE_EXPORT void*
  _AddVariableChangedDelegate(VariableRef* var,void (*func)());
  extern "C" ARCANE_CORE_EXPORT void
  _RemoveVariableChangedDelegate(VariableRef::UpdateNotifyFunctorList* functor_list,
                                 void (*func)());
  extern "C" ARCANE_EXPORT void
  _ArcaneWrapperCoreAddVariableChangedDelegate(VariableRef* var,void (*func)())
  {
    _AddVariableChangedDelegate(var,func);
  }
  extern "C" ARCANE_EXPORT void
  _ArcaneWrapperCoreRemoveVariableChangedDelegate(VariableRef::UpdateNotifyFunctorList* var,void (*func)())
  {
    _RemoveVariableChangedDelegate(var,func);
  }
%}

%typemap(csdisposing_derived, methodname="Dispose", methodmodifiers="public") Arcane::MeshVariableRef ""

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  template<typename DataType> class ItemVariableArrayRefT;
  template<typename DataType> class ItemVariableScalarRefT;
  template<typename DataType> class VariableRefScalarT;

  template<typename ItemType,typename DataType> class MeshVariableArrayRefT;
  class MeshVariableRef : public VariableRef
  {
   public:
    ~MeshVariableRef() {}
    virtual Integer arraySize() const = 0;
    void synchronize();
#if defined(SWIGCSHARP)
    %proxycode
%{
    protected override void Dispose(bool is_disposing)
    {
      is_disposed = true;
      // Il faut faire tres attention dans le dispose de ne pas
      // utiliser de variables C++ qui peuvent avoir ete desallouees
      lock(this) {
        if (swigCPtr.Handle != IntPtr.Zero) {
          //Console.WriteLine("DISPOSE VAR del={0} ptr={1}",saved_del,m_notify_ptr);
          if (m_update_notifier!=null){
            m_update_notifier.Unregister();
            saved_del_handle.Free();
            m_update_notifier = null;
          }
          if (swigCMemOwn) {
            swigCMemOwn = false;
            $imclassname.delete_MeshVariableRef(swigCPtr);
          }
          swigCPtr = new HandleRef(null, IntPtr.Zero);
        }
        base.Dispose(is_disposing);
      }
    }
%}
#endif
  };
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_VARIABLE_ITEMSCALAR_DEFINE(DATATYPE, FULLCTYPE )
%typemap(csinterfaces) Arcane::ItemVariableScalarRefT< DATATYPE > "";
%typemap(csbody_derived) Arcane::ItemVariableScalarRefT< DATATYPE >
%{
  private HandleRef swigCPtr;

  internal $csclassname(IntPtr cPtr, bool cMemoryOwn) : base($imclassname.VariableItem##DATATYPE##_SWIGUpcast(cPtr), cMemoryOwn) {
    swigCPtr = new HandleRef(this, cPtr);
    _InitMeshVariable();
  }

  internal static HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new HandleRef(null, IntPtr.Zero) : obj.swigCPtr;
  }
%}
%typemap(csclassmodifiers) ItemVariableScalarRefT< DATATYPE > "public unsafe class"
%typemap(cscode) ItemVariableScalarRefT< DATATYPE > %{
  private DATATYPE * m_values;
  private Integer m_size;
  private DATATYPE##ArrayView m_view;

  protected void _InitMeshVariable()
  {
    _InitMeshVariableBase();
    DATATYPE##ArrayView v = this._asArray();
    m_view = v;
    m_values = v._InternalData();
    m_size = v.Size;
    //Console.WriteLine("BUILD ARRAY name={0} size={1} ptr={2}",name(),m_size,(IntPtr)m_values);
  }
  public DATATYPE this[Integer index]
  {
    get
    {
      return m_values[index];
    }
    set
    {
      m_values[index] = value;
    }
  }
  public DATATYPE##ArrayView AsArray()
  {
    return m_view;
  }
  public DATATYPE this[Item item]
  {
    get { return m_values[item.LocalId]; }
    set { m_values[item.LocalId] = value; }
  }
  public DATATYPE this[IItem item]
  {
    get { return m_values[item.LocalId]; }
    set { m_values[item.LocalId] = value; }
  }
  protected override void _OnSizeChanged()
   {
		 DATATYPE##ArrayView v = _asArray();
     m_view = v;
     m_values = v._InternalData();
     m_size = v.Size;
     //Console.WriteLine("GET ARRAY name={0} size={1} ptr={2}",Name,m_size,(IntPtr)m_values);
   }
	 %}
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_VARIABLE_ITEM_SPECIALIZE(DATATYPE )
namespace Arcane
{
  template<> class ItemVariableScalarRefT<DATATYPE >
    : public MeshVariableRef
  {
  public:
    ItemVariableScalarRefT(const VariableBuildInfo& b,eItemKind ik);
   protected:
   // TODO SUPPRIMER PUBLIC ET LAISSER PROTECTED MAIS IL FAUT QUE
   // SWIG GENERE QUAND MEME LES FONCTIONS
   public:
    void fill(DATATYPE v);
    //void copy(ConstArrayView<DATATYPE> v);
    //TODO: faire la copie dans la classe Mesh pour eviter de copier une variable d'un autre genre.
    void copy(const ItemVariableScalarRefT<DATATYPE>& v);
    ArrayView<DATATYPE> asArray();
    void updateFromInternal();
    virtual Integer arraySize() const;
    ItemGroup itemGroup() const;
    SWIG_ARCANE_VARIABLE_ITEMSCALAR_DEFINE(DATATYPE ,VariableItem##DATATYPE )
  };
  template<> class ItemVariableArrayRefT< DATATYPE >
    : public MeshVariableRef
  {
  public:
    SWIG_ARCANE_VARIABLE_ITEMARRAY_DEFINE(DATATYPE ,VariableItemArray##DATATYPE )
      ItemVariableArrayRefT(const VariableBuildInfo& b,eItemKind ik);
   protected:
    // TODO SUPPRIMER PUBLIC ET LAISSER PROTECTED MAIS IL FAUT QUE
    // SWIG GENERE QUAND MEME LES FONCTIONS
  public:
    Array2View<DATATYPE> asArray();
    void updateFromInternal();
    virtual bool isArrayVariable() const;
    virtual Integer arraySize() const;
    void resize(Integer s);
    ItemGroup itemGroup() const;
  };
}
%template(VariableItem##DATATYPE) Arcane::ItemVariableScalarRefT< DATATYPE >;
%template(VariableItemArray##DATATYPE) Arcane::ItemVariableArrayRefT< DATATYPE >;
%enddef


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Implementation pour une variable scalaire sur une entite du maillage
%define SWIG_ARCANE_VARIABLE_MESHSCALAR_DEFINE(DATATYPE, ITEMTYPE )
%typemap(cscode) Arcane::MeshVariableScalarRefT<ITEMTYPE ,DATATYPE >
%{
  public DATATYPE this[ITEMTYPE item]
  {
    get { return base[item.LocalId]; }
    set { base[item.LocalId] = value; }
  }
  public Variable##ITEMTYPE##DATATYPE _Internal { get { return this; } }
%}
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  template<typename ItemType,typename DataType> class MeshVariableScalarRefT;
}

%define SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE2(ITEMTYPE, DATATYPE )
namespace Arcane
{
  template<> class MeshVariableScalarRefT<ITEMTYPE ,DATATYPE >
    : public ItemVariableScalarRefT<DATATYPE >
 {
 public:
   MeshVariableScalarRefT(const VariableBuildInfo& vbi);
   SWIG_ARCANE_VARIABLE_MESHSCALAR_DEFINE(DATATYPE ,ITEMTYPE)
  };

  template<> class MeshVariableArrayRefT<ITEMTYPE ,DATATYPE >
    : public ItemVariableArrayRefT<DATATYPE>
  {
  public:
    MeshVariableArrayRefT(const VariableBuildInfo& sbi);
    SWIG_ARCANE_VARIABLE_MESHARRAY_DEFINE(DATATYPE ,ITEMTYPE)
  };
 }

// Il faut mettre les typedef pour être sur que swig génère toujours
// correctement les bons wrappers.
typedef Arcane::MeshVariableScalarRefT<ITEMTYPE, DATATYPE> Variable##ITEMTYPE##DATATYPE;
typedef Arcane::MeshVariableArrayRefT<ITEMTYPE,DATATYPE> Variable##ITEMTYPE##Array##DATATYPE;
%template(Variable##ITEMTYPE##DATATYPE) Arcane::MeshVariableScalarRefT<ITEMTYPE, DATATYPE >;
%template(Variable##ITEMTYPE##Array##DATATYPE) Arcane::MeshVariableArrayRefT<ITEMTYPE, DATATYPE >;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Implementation pour une variable tableau sur une entite du maillage
%define SWIG_ARCANE_VARIABLE_MESHARRAY_DEFINE(DATATYPE, ITEMTYPE)
%typemap(cscode) Arcane::MeshVariableArrayRefT<ITEMTYPE ,DATATYPE > %{
  public DATATYPE##ArrayView this[ITEMTYPE item]
  {
    get { return base[item.LocalId]; }
  }
%}
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_VARIABLE_ITEMARRAY_DEFINE(DATATYPE, FULLCTYPE )
%typemap(csbody_derived) ItemVariableArrayRefT<DATATYPE >
%{
  private HandleRef swigCPtr;

  internal $csclassname(IntPtr cPtr, bool cMemoryOwn) : base($imclassname.VariableItemArray##DATATYPE##_SWIGUpcast(cPtr), cMemoryOwn){
    swigCPtr = new HandleRef(this, cPtr);
    _InitMeshVariable();
  }

  internal static HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new HandleRef(null, IntPtr.Zero) : obj.swigCPtr;
  }
%}
%typemap(csclassmodifiers) ItemVariableArrayRefT<DATATYPE > "public unsafe class"
%typemap(cscode) ItemVariableArrayRefT< DATATYPE > %{
  private DATATYPE##Array2View m_values;

  protected void _InitMeshVariable()
  {
    _InitMeshVariableBase();
    m_values = _asArray();
  }

  public DATATYPE##ArrayView this[Integer index]
  {
    get { return m_values[index]; }
  }

  public DATATYPE##ArrayView this[Item item]
  {
    get { return m_values[item.LocalId]; }
  }
  public DATATYPE##ArrayView this[IItem item]
  {
    get { return m_values[item.LocalId]; }
  }
  protected override void _OnSizeChanged()
   {
     m_values = _asArray();
     //Console.WriteLine("GET ARRAY (ItemArray) name={0} size1={1} size2={2} ptr={3}",Name,
     //                  m_values.Dim1Size,m_values.Dim2Size,(IntPtr)m_values._Base());
   }
%}
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(DATATYPE )
SWIG_ARCANE_VARIABLE_ITEM_SPECIALIZE(DATATYPE )
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE2(Node, DATATYPE )
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE2(Face, DATATYPE )
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE2(Cell, DATATYPE )
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Effectue le wrapping des variables
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Real)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Int16)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Int32)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Int64)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Real3)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Real3x3)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Real2)
SWIG_ARCANE_VARIABLE_MESH_SPECIALIZE(Real2x2)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * GESTION DES VARIABLES SCALAIRES
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_VARIABLE_SCALAR_DEFINE(DATATYPE, FULLCTYPE )
%typemap(csinterfaces) Arcane::VariableRefScalarT< DATATYPE > "";
%typemap(cscode) Arcane::VariableRefScalarT< DATATYPE >
%{
  public DATATYPE Value
  {
    get { return _value(); }
    set { Assign(value); }
  }
%}
%enddef

%define SWIG_ARCANE_VARIABLE_SCALAR_SPECIALIZE(DATATYPE )
namespace Arcane
{
  template<> class VariableRefScalarT<DATATYPE > : public VariableRef
  {
  public:
    VariableRefScalarT(const VariableBuildInfo& b);
  protected:
    // TODO SUPPRIMER PUBLIC ET LAISSER PROTECTED MAIS IL FAUT QUE
    // SWIG GENERE QUAND MEME LES FONCTIONS
  public:
    virtual Integer arraySize() const;
    const DATATYPE& value() const;
    void assign(const DATATYPE& v);
    SWIG_ARCANE_VARIABLE_SCALAR_DEFINE(DATATYPE ,VariableScalar##DATATYPE )
  };
}
%template(VariableScalar##DATATYPE) Arcane::VariableRefScalarT< DATATYPE >;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SWIG_ARCANE_VARIABLE_SCALAR_SPECIALIZE(Real)
SWIG_ARCANE_VARIABLE_SCALAR_SPECIALIZE(Real3)
SWIG_ARCANE_VARIABLE_SCALAR_SPECIALIZE(Int16)
SWIG_ARCANE_VARIABLE_SCALAR_SPECIALIZE(Int32)
SWIG_ARCANE_VARIABLE_SCALAR_SPECIALIZE(Int64)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
