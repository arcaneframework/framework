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

%include arcane/core/materials/ComponentPartItemVectorView.h
%include arcane/core/materials/ComponentItemVectorView.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
