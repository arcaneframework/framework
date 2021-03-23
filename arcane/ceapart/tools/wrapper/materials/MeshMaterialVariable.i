/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%include arcane/materials/IMeshMaterialVariable.h
%include arcane/materials/MeshMaterialVariableRef.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(DATATYPE)
%typemap(csbody_derived) Arcane::Materials::CellMaterialVariableScalarRef< DATATYPE >
%{
  private HandleRef swigCPtr;

  internal $csclassname(IntPtr cPtr, bool cMemoryOwn) : base($imclassname.MaterialVariableCell##DATATYPE##_SWIGUpcast(cPtr), cMemoryOwn)
  {
    swigCPtr = new HandleRef(this, cPtr);
    _InitMeshVariable();
  }

  internal static HandleRef getCPtr($csclassname obj)
  {
    return (obj == null) ? new HandleRef(null, IntPtr.Zero) : obj.swigCPtr;
  }
%}
%typemap(csclassmodifiers) Arcane::Materials::CellMaterialVariableScalarRef< DATATYPE > "public unsafe class"
%typemap(cscode) Arcane::Materials::CellMaterialVariableScalarRef< DATATYPE >
%{
  DATATYPE##ArrayView* m_values;

  protected void _InitMeshVariable()
  {
    m_values = (DATATYPE##ArrayView*)_internalValueAsPointer();
  }
  public DATATYPE this[MatVarIndex item]
  {
    get { return m_values[item.ArrayIndex][item.ValueIndex]; }
    set { m_values[item.ArrayIndex][item.ValueIndex] = value; }
  }
  public DATATYPE this[ComponentItem item]
  {
    get { return m_values[item._matvarArrayIndex][item._matvarValueIndex]; }
    set { m_values[item._matvarArrayIndex][item._matvarValueIndex] = value; }
  }
  protected void _OnSizeChanged()
  {
    m_values = (DATATYPE##ArrayView*)_internalValueAsPointer();
  }
%}

%template(MaterialVariableCell##DATATYPE) Arcane::Materials::CellMaterialVariableScalarRef< DATATYPE >;

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Real)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Int16)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Int32)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Int64)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Real2)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Real3)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Real2x2)
ARCANE_SWIG_MATERIAL_VARIABLE_SCALAR_CELL(Real3x3)
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
