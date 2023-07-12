// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapper pour les options du jeu de données.
//
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  template<typename DataType> class CaseOptionMultiSimpleT;
  template<typename DataType> class CaseOptionSimpleT;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%feature("director") CaseOptionEnum;
%feature("director") CaseOptionMultiEnum;
%feature("director") CaseOptionExtended;
%feature("director") CaseOptionMultiExtended;
%feature("director") CaseOptionsMulti;
%feature("director") Arcane::ICaseOptionServiceContainer;
%feature("director") Arcane::StandardCaseFunction;
%feature("director") Arcane::CaseFunction;
%feature("director") Arcane::CaseFunction2;
%feature("director") Arcane::CaseOptionComplexValue;

%ignore Arcane::ICaseFunction::value;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Supprime dans les classes CaseOption* avec director l'appel
// au delete C++ dans le Dispose C#. En effet, les options
// sont gérées directement par le 'CaseOptions' et ne
// doivent donc pas être détruite à la main.

%typemap(csdisposing, methodname="Dispose", methodmodifiers="public") Arcane::ICaseOptions ""
%typemap(csdispose, methodname="Dispose", methodmodifiers="public") Arcane::ICaseOptions %{ public virtual void Dispose() {} %}
%typemap(csdispose_derived) Arcane::CaseOptions ""
%typemap(csdispose_derived) Arcane::CaseOptionsMulti ""
%typemap(csdisposing_derived) Arcane::CaseOptions ""
%typemap(csdisposing_derived) Arcane::CaseOptionsMulti ""
%typemap(csdisposing_derived) Arcane::CaseOptionMultiServiceImpl ""
%typemap(csdisposing_derived) Arcane::CaseOptionServiceImpl ""
%typemap(csdispose, methodname="Dispose", methodmodifiers="public") Arcane::CaseOptionService  %{ public virtual void Dispose() {} %}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%include arcane/core/CaseOptionTypes.h
%include arcane/core/CaseOptionBuildInfo.h
%include arcane/core/CaseOptionBase.h
%include arcane/core/ICaseOptions.h
%include arcane/core/ICaseOptionList.h
%include arcane/core/CaseOptionComplexValue.h
%include arcane/core/CaseOptionSimple.h
%include arcane/core/CaseOptionEnum.h
%include arcane/core/CaseOptionExtended.h
%include arcane/core/CaseOptions.h
%include arcane/core/ICaseMng.h
%include arcane/core/ICaseFunction.h
%include arcane/core/CaseFunction.h
%include arcane/core/StandardCaseFunction.h
%include arcane/core/CaseFunction2.h
%include arcane/core/ICaseFunctionProvider.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Redéfini le CaseOptionsMulti en supprimant l'héritage multiple
namespace Arcane
{
 class CaseOptionsMulti : public CaseOptions
 {
  public:
	
   CaseOptionsMulti(ICaseMng*,const String& tag_root_name,
                    const XmlNode& element,Integer min_occurs,Integer max_occurs);
   CaseOptionsMulti(ICaseOptionList*,const String& tag_root_name,
                    const XmlNode& element,Integer min_occurs,Integer max_occurs);

  protected:
     // Pour le C#, il ne faut pas appeler le destructeur de cette option
     ~CaseOptionsMulti(){}

  public:

   virtual void multiAllocate(const XmlNodeList&) =0;
   virtual ICaseOptions* toCaseOptions() { return this; }
   ICaseOptionsMulti* toCaseOptionsMulti() { return this; }
 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%include arcane/core/CaseOptionServiceImpl.h

 /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  template<class T>
  class CaseOptionExtendedT : public CaseOptionExtended
  {
   public:
    CaseOptionExtendedT(const CaseOptionBuildInfo& cob,const String& type_name)
    : CaseOptionExtended(cob,type_name) {}
  };

  template<class T>
  class CaseOptionMultiExtendedT : public CaseOptionMultiExtended
  {
   public:
    CaseOptionMultiExtendedT(const CaseOptionBuildInfo& cob,const String& type_name)
    : CaseOptionMultiExtended(cob,type_name) {}
  };

  template<class T>
  class CaseOptionMultiSimpleT : public CaseOptionMultiSimple
  {
   public:
    CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob);
    CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob,const String& physical_unit);
   public:
    ConstArrayView<T> values() const;
    const T& value(Integer index) const;
    Integer size() const;
   protected:
    // Il faut redefinir ces 5 methodes sinon SWIG considere que cette classe
    // n'est pas instantiable
    virtual void print(const String& lang,std::ostream& o) const;
    virtual ICaseFunction* function() const { return 0; }
    virtual void updateFromFunction(Real current_time,Integer current_iteration) {}
    virtual void _search(bool is_phase1);
    virtual void visit(ICaseDocumentVisitor*) const;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_CASEOPTION_EXTENDED(ITEMTYPE,FAMILYNAME)
%typemap(csbase) Arcane::CaseOptionExtendedT<ITEMTYPE > "Arcane.CaseOptionExtended"
%typemap(csbody) Arcane::CaseOptionExtendedT<ITEMTYPE > %{ %}
%typemap(SWIG_DISPOSING) Arcane::CaseOptionExtendedT<ITEMTYPE > ""
%typemap(SWIG_DISPOSE, methodname="Dispose", methodmodifiers="private") CaseOptionExtendedT<ITEMTYPE > ""
%typemap(csin) Arcane::CaseOptionExtendedT<ITEMTYPE > "$csinput"

%typemap(cscode) Arcane::CaseOptionExtendedT<ITEMTYPE >
%{
    ITEMTYPE m_value;

    public $csclassname(CaseOptionBuildInfo cob,string type_name)
    : base(cob,type_name)
    {
    }

    public ITEMTYPE Value { get { return m_value; } }

    public ITEMTYPE value() { return m_value; }

    protected override bool _tryToConvert(string s)
    {
      object v = _ConvertToItemGroup(s);
      m_value = (ITEMTYPE)v;
      return (v==null);
    }
    private object _ConvertToItemGroup(string name)
    {
      ISubDomain sub_domain = CaseMng().SubDomain();
      IMesh mesh = sub_domain.DefaultMesh();
      ItemGroup item_group = mesh.FAMILYNAME().FindGroup(name);
      if (item_group.IsNull())
        return null;
      ITEMTYPE true_group = new ITEMTYPE(item_group);
      if (true_group.IsNull())
        return null;
      return true_group;
  }
%}

%template(ITEMTYPE##ExtendedCaseOption) Arcane::CaseOptionExtendedT<ITEMTYPE>;

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_CASEOPTION_MULTIEXTENDED(ITEMTYPE,FAMILYNAME)
%typemap(csbase) Arcane::CaseOptionMultiExtendedT<ITEMTYPE > "Arcane.CaseOptionMultiExtended"
%typemap(csbody) Arcane::CaseOptionMultiExtendedT<ITEMTYPE > %{ %}
%typemap(SWIG_DISPOSING) Arcane::CaseOptionMultiExtendedT<ITEMTYPE > ""
%typemap(csdispose, methodname="Dispose", methodmodifiers="private") CaseOptionMultiExtendedT<ITEMTYPE > ""
%typemap(csin) Arcane::CaseOptionMultiExtendedT<ITEMTYPE > "$csinput"

%typemap(cscode) Arcane::CaseOptionMultiExtendedT<ITEMTYPE >
%{
    ITEMTYPE[] m_values;

    public $csclassname(CaseOptionBuildInfo cob,string type_name)
    : base(cob,type_name)
    {
    }

    public ITEMTYPE[] Values { get { return m_values; } }

    protected override void _allocate(Integer n)
    {
      m_values = new ITEMTYPE[n];
    }

    protected override Integer _nbElem()
    {
      if (m_values==null)
        return 0;
      return m_values.Length;
    }

    protected override bool _tryToConvert(string s,Integer index)
    {
      object v = _ConvertToItemGroup(s);
      m_values[index] = (ITEMTYPE)v;
      return (v==null);
    }

    private object _ConvertToItemGroup(string name)
    {
      ISubDomain sub_domain = CaseMng().SubDomain();
      IMesh mesh = sub_domain.DefaultMesh();
      ItemGroup item_group = mesh.FAMILYNAME().FindGroup(name);
      if (item_group.IsNull())
        return null;
      ITEMTYPE true_group = new ITEMTYPE(item_group);
      if (true_group.IsNull())
        return null;
      return true_group;
  }
%}

%template(ITEMTYPE##MultiExtendedCaseOption) Arcane::CaseOptionMultiExtendedT<ITEMTYPE>;

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(csbase) Arcane::CaseOptionExtendedT<ItemGroup> "Arcane.CaseOptionExtended"
%typemap(csbody) Arcane::CaseOptionExtendedT<ItemGroup> %{ %}
%typemap(csdisposing) Arcane::CaseOptionExtendedT<ItemGroup> ""
%typemap(csdispose, methodname="Dispose", methodmodifiers="private") CaseOptionExtendedT<ItemGroup> ""
%typemap(csin) Arcane::CaseOptionExtendedT<ItemGroup > "$csinput"
%typemap(cscode) Arcane::CaseOptionExtendedT<ItemGroup >
%{
    ItemGroup m_value;

    public $csclassname(CaseOptionBuildInfo cob,string type_name)
    : base(cob,type_name)
    {
    }

    public ItemGroup Value { get { return m_value; } }

    public ItemGroup value() { return m_value; }

    protected override bool _tryToConvert(string s)
    {
      object v = _ConvertToItemGroup(s);
      m_value = (ItemGroup)v;
      return (v==null);
    }
    private object _ConvertToItemGroup(string name)
    {
      ISubDomain sub_domain = CaseMng().SubDomain();
      IMesh mesh = sub_domain.DefaultMesh();
      ItemGroup item_group = mesh.FindGroup(name);
      if (item_group.IsNull())
        return null;
      return item_group;
  }
%}

%template(ItemGroupExtendedCaseOption) Arcane::CaseOptionExtendedT<ItemGroup>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(csbase) Arcane::CaseOptionMultiExtendedT<ItemGroup> "Arcane.CaseOptionMultiExtended"
%typemap(csbody) Arcane::CaseOptionMultiExtendedT<ItemGroup> %{ %}
%typemap(csdisposing) Arcane::CaseOptionMultiExtendedT<ItemGroup> ""
%typemap(csdispose, methodname="Dispose", methodmodifiers="private") CaseOptionMultiExtendedT<ItemGroup > ""
%typemap(csin) Arcane::CaseOptionMultiExtendedT<ItemGroup> "$csinput"
%typemap(cscode) Arcane::CaseOptionMultiExtendedT<ItemGroup>
%{
    ItemGroup[] m_values;

    public $csclassname(CaseOptionBuildInfo cob,string type_name)
    : base(cob,type_name)
    {
    }

    public ItemGroup[] Values { get { return m_values; } }

    protected override void _allocate(Integer n)
    {
      m_values = new ItemGroup[n];
    }

    protected override Integer _nbElem()
    {
      if (m_values==null)
        return 0;
      return m_values.Length;
    }

    protected override bool _tryToConvert(string s,Integer index)
    {
      object v = _ConvertToItemGroup(s);
      m_values[index] = (ItemGroup)v;
      return (v==null);
    }

    private object _ConvertToItemGroup(string name)
    {
      ISubDomain sub_domain = CaseMng().SubDomain();
      IMesh mesh = sub_domain.DefaultMesh();
      ItemGroup item_group = mesh.FindGroup(name);
      if (item_group.IsNull())
        return null;
      return item_group;
  }
%}

%template(ItemGroupMultiExtendedCaseOption) Arcane::CaseOptionMultiExtendedT<ItemGroup>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_CASEOPTION_MULTISIMPLE(DATATYPE)
%typemap(cscode) Arcane::CaseOptionMultiSimpleT<DATATYPE >
%{
  public DATATYPE[] Values { get { return _values().ToArray(); } }
%}
%template(CaseOptionMultiSimple##DATATYPE) Arcane::CaseOptionMultiSimpleT<DATATYPE>;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/// Implementation speciale pour les 'string'
%typemap(cscode) Arcane::CaseOptionMultiSimpleT<String>
%{
  public string[] Values
  {
    get
    {
      Integer n = Size();
      string[] v = new string[n];
      for( Integer i=0; i<n; ++i )
        v[i] = Value(i);
      return v;
    }
  }
%}
%template(CaseOptionMultiSimpleString) Arcane::CaseOptionMultiSimpleT<Arcane::String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/// Implementation speciale pour les types simples
%typemap(cscode) Arcane::CaseOptionSimpleT<Real>
%{
  public static implicit operator Real(CaseOptionReal v)
  {
    return v.Value();
  }
%}
%typemap(cscode) Arcane::CaseOptionSimpleT<Int32>
%{
  public static implicit operator Int32(CaseOptionInt32 v)
  {
    return v.Value();
  }
%}
%typemap(cscode) Arcane::CaseOptionSimpleT<bool>
%{
  public static implicit operator bool(CaseOptionBool v)
  {
    return v.Value();
  }
%}
%typemap(cscode) Arcane::CaseOptionSimpleT<Real3>
%{
  public static implicit operator Real3(CaseOptionReal3 v)
  {
    return v.Value();
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%template(CaseOptionReal) Arcane::CaseOptionSimpleT<Real>;
%template(CaseOptionReal2) Arcane::CaseOptionSimpleT<Real2>;
%template(CaseOptionReal3) Arcane::CaseOptionSimpleT<Real3>;
%template(CaseOptionReal2x2) Arcane::CaseOptionSimpleT<Real2x2>;
%template(CaseOptionReal3x3) Arcane::CaseOptionSimpleT<Real3x3>;
%template(CaseOptionInt32) Arcane::CaseOptionSimpleT<Int32>;
%template(CaseOptionInt64) Arcane::CaseOptionSimpleT<Int64>;
%template(CaseOptionBool) Arcane::CaseOptionSimpleT<bool>;
%template(CaseOptionString) Arcane::CaseOptionSimpleT<String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SWIG_ARCANE_CASEOPTION_EXTENDED(CellGroup,CellFamily)
SWIG_ARCANE_CASEOPTION_EXTENDED(FaceGroup,FaceFamily)
SWIG_ARCANE_CASEOPTION_EXTENDED(NodeGroup,NodeFamily)
SWIG_ARCANE_CASEOPTION_EXTENDED(EdgeGroup,EdgeFamily)

SWIG_ARCANE_CASEOPTION_MULTIEXTENDED(CellGroup,CellFamily)
SWIG_ARCANE_CASEOPTION_MULTIEXTENDED(FaceGroup,FaceFamily)
SWIG_ARCANE_CASEOPTION_MULTIEXTENDED(NodeGroup,NodeFamily)
SWIG_ARCANE_CASEOPTION_MULTIEXTENDED(EdgeGroup,EdgeFamily)

SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Int32)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Int64)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(String)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Real)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Real3)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Real3x3)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Real2)
SWIG_ARCANE_CASEOPTION_MULTISIMPLE(Real2x2)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(csbase) Arcane::CaseOptionServiceT<AbstractService> "Arcane.CaseOptionServiceImpl"
%typemap(csbody) Arcane::CaseOptionServiceT<AbstractService> %{ %}
%typemap(csdisposing) Arcane::CaseOptionServiceT<AbstractService> ""
%typemap(csdispose, methodname="Dispose", methodmodifiers="private") CaseOptionServiceT<AbstractService> ""
%typemap(csin) Arcane::CaseOptionExtendedT<AbstractService> "$csinput"
%typemap(cscode) Arcane::CaseOptionExtendedT<AbstractService>
%{
    ItemGroup m_value;

    public $csclassname(CaseOptionBuildInfo cob,string type_name)
    : base(cob,type_name)
    {
    }

    public ItemGroup Value { get { return m_value; } }

    public ItemGroup value() { return m_value; }

    protected override bool _tryToConvert(string s)
    {
      object v = _ConvertToItemGroup(s);
      m_value = (ItemGroup)v;
      return (v==null);
    }
    private object _ConvertToItemGroup(string name)
    {
      ISubDomain sub_domain = CaseMng().SubDomain();
      IMesh mesh = sub_domain.DefaultMesh();
      ItemGroup item_group = mesh.FindGroup(name);
      if (item_group.IsNull())
        return null;
      return item_group;
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%template(CaseFunctionRef) Arcane::Ref<Arcane::ICaseFunction>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
