/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Materials::ComponentItemInternalPtr,ComponentItemInternalPtr)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Materials::MatVarIndex,MatVarIndex)

ARCANE_SWIG_SPECIALIZE_CONSTARRAYVIEW(Arcane::Materials::IMeshComponentPtr,Arcane.Materials.MeshComponentListView)
ARCANE_SWIG_SPECIALIZE_CONSTARRAYVIEW(Arcane::Materials::IMeshMaterialPtr,Arcane.Materials.MeshMaterialListView)
ARCANE_SWIG_SPECIALIZE_CONSTARRAYVIEW(Arcane::Materials::IMeshEnvironmentPtr,Arcane.Materials.MeshEnvironmentListView)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(cscode) Arcane::Materials::IMeshComponent
%{
  public ComponentItemEnumerator<ComponentItem> GetEnumerator()
  {
    return ComponentItemEnumeratorBuilder.Create(this, new ComponentItem());
  }
%}

%typemap(cscode) Arcane::Materials::IMeshEnvironment
%{
  public new ComponentItemEnumerator<EnvItem> GetEnumerator()
  {
    return ComponentItemEnumeratorBuilder.Create(this, new EnvItem());
  }
%}

%typemap(cscode) Arcane::Materials::IMeshMaterial
%{
  public new ComponentItemEnumerator<MatItem> GetEnumerator()
  {
    return ComponentItemEnumeratorBuilder.Create(this, new MatItem());
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define ARCANE_SWIG_MATERIAL_COMPONENTITEMVECTORVIEW(CTYPE,CSHARP_TYPE)

%typemap(cstype) CTYPE %{ CSHARP_TYPE %}
%typemap(imtype) CTYPE %{ CSHARP_TYPE %}
%typemap(csin) CTYPE "$csinput"
%typemap(csout) CTYPE {
    CSHARP_TYPE ret = $imcall;$excode
    return ret;
  }
%typemap(ctype, out="Arcane::Materials::ComponentItemVectorViewPOD") CTYPE %{ CSHARP_TYPE %}
%typemap(out) CTYPE
%{
   Arcane::Materials::ComponentItemVectorView result_ref = ($1);
   $result = _createComponentItemVectorViewPOD(result_ref);
%}
%typemap(in) CTYPE %{$1 = $input; %}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_SWIG_MATERIAL_COMPONENTITEMVECTORVIEW(Arcane::Materials::ComponentItemVectorView,Arcane.Materials.ComponentItemVectorView)
ARCANE_SWIG_MATERIAL_COMPONENTITEMVECTORVIEW(Arcane::Materials::MatItemVectorView,Arcane.Materials.MatItemVectorView)
ARCANE_SWIG_MATERIAL_COMPONENTITEMVECTORVIEW(Arcane::Materials::EnvItemVectorView,Arcane.Materials.EnvItemVectorView)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%include arcane/core/materials/ComponentPartItemVectorView.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
