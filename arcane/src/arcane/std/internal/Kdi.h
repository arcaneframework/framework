// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Kdi.h                                                       (C) 2000-2025 */
/*                                                                           */
/* Post-traitement avec l'outil KDI.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_KDI_H
#define ARCANE_STD_INTERNAL_KDI_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define Py_LIMITED_API 0x03100000
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_23_API_VERSION
#include <Python.h>

#include <numpy/arrayobject.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#define KASSERT(cnd) \
  if (!(cnd)) { \
    ARCANE_FATAL("KDI ASSERT"); \
  }
#define KTRACE(trace, msg) \
  if (trace) { \
    std::cout << msg << std::endl; \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KDIChunk
{
 private:

  const PyObject* m_pInstanceChunk;
  const bool m_trace;

 public:

  KDIChunk(const PyObject* _pInstanceChunk, bool _trace = false)
  : m_pInstanceChunk(_pInstanceChunk)
  , m_trace(_trace)
  {
    KTRACE(m_trace, "KDIChunk::KDIChunk IN/OUT");
  }

  ~KDIChunk()
  {
    KTRACE(m_trace, "KDIChunk::~KDIChunk IN");
    if (m_pInstanceChunk)
      Py_DECREF(m_pInstanceChunk);
    KTRACE(m_trace, "KDIChunk::~KDIChunk OUT");
  }

 private:

  PyObject* _simple_call(const std::string _method, PyObject* _pArgs)
  {
    KTRACE(m_trace, "KDIChunk:_simple_call IN");
    PyObject* pValue = PyUnicode_FromString("pykdi");
    KASSERT(pValue);
    PyObject* pModule = PyImport_Import(pValue);
    KASSERT(pModule);
    PyObject* pClass = PyObject_GetAttrString(pModule, "KDIAgreementStepPartChunk");
    KASSERT(pClass);
    PyObject* pMethod = PyObject_GetAttrString(pClass, _method.c_str());
    KASSERT(pMethod);
    KASSERT(PyCallable_Check(pMethod));
    KASSERT(_pArgs);
    PyObject* pResult = PyObject_CallObject(pMethod, _pArgs);
    Py_DECREF(pMethod);
    Py_DECREF(pClass);
    Py_DECREF(pModule);
    Py_DECREF(pValue);
    KTRACE(m_trace, "KDIChunk:_simple_call OUT");
    return pResult;
  }

 public:

  void dump()
  {
    KTRACE(m_trace, "KDIChunk:dump IN");
    KASSERT(m_pInstanceChunk);
    PyObject* pArgs = Py_BuildValue("(O)", m_pInstanceChunk);
    _simple_call("dump", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIChunk:dump OUT");
  }

  void set(const std::string _absname, PyArrayObject* _array)
  {
    KTRACE(m_trace, "KDIChunk:set IN");
    KASSERT(m_pInstanceChunk);
    PyObject* pArgs = Py_BuildValue("(O, z, O)", m_pInstanceChunk, _absname.c_str(), _array);
    _simple_call("set", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIChunk:set OUT");
  }

  void saveVTKHDF(const std::string _absfilename)
  {
    // TODO On pourrait vérifier que l'on est partless au moins en 0.4.0
    KTRACE(m_trace, "KDIChunk:saveVTKHDF (...) IN");
    // TODO Verifier que l'on est bien partless
    KASSERT(m_pInstanceChunk);
    PyObject* pArgs = Py_BuildValue("(O, z)", m_pInstanceChunk, _absfilename.c_str());
    KASSERT(pArgs);
    _simple_call("saveVTKHDF", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIChunk:saveVTKHDF OUT");
  }

  void saveVTKHDFCompute(const std::string _absfilename)
  {
    KTRACE(m_trace, "KDI KDIChunk saveVTKHDFCompute " << _absfilename);
    // TODO On pourrait vérifier que l'on est partless au moins en 0.4.0
    // TODO ON pourrait trouver une facon plus elegante d'activté des traitements ici KDIComputeMultiMilieux
    KTRACE(m_trace, "KDIChunk:saveVTKHDFCompute (...) IN");
    // TODO Verifier que l'on est bien partless
    KASSERT(m_pInstanceChunk);
    PyObject* pArgs = Py_BuildValue("(O, z)", m_pInstanceChunk, _absfilename.c_str());
    KASSERT(pArgs);
    _simple_call("saveVTKHDFCompute", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDI KDIChunk saveVTKHDFCompute");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KDIBase
{
 private:

  const PyObject* m_pInstanceBase;
  const bool m_trace;

 public:

  KDIBase(const PyObject* _pInstanceBase, const bool _trace)
  : m_pInstanceBase(_pInstanceBase)
  , m_trace(_trace)
  {
    KTRACE(m_trace, "KDIBase::KDIBase IN/OUT");
  }

  ~KDIBase()
  {
    KTRACE(m_trace, "KDIBase::~KDIBase IN");
    if (m_pInstanceBase)
      Py_DECREF(m_pInstanceBase);
    KTRACE(m_trace, "KDIBase::~KDIBase OUT");
  }

 private:

  PyObject* _simple_call(const std::string _method, PyObject* _pArgs)
  {
    KTRACE(m_trace, "KDIBase:_simple_call IN");
    PyObject* pValue = PyUnicode_FromString("pykdi");
    KASSERT(pValue);
    PyObject* pModule = PyImport_Import(pValue);
    KASSERT(pModule);
    PyObject* pClass = PyObject_GetAttrString(pModule, "KDIAgreementStepPartBase");
    KASSERT(pClass);
    PyObject* pMethod = PyObject_GetAttrString(pClass, _method.c_str());
    KASSERT(pMethod);
    KASSERT(PyCallable_Check(pMethod));
    KASSERT(_pArgs);
    PyObject* pResult = PyObject_CallObject(pMethod, _pArgs);
    Py_DECREF(pMethod);
    Py_DECREF(pClass);
    Py_DECREF(pModule);
    Py_DECREF(pValue);
    KTRACE(m_trace, "KDIBase:_simple_call OUT");
    return pResult;
  }

 public:

  void dump()
  {
    KTRACE(m_trace, "KDIBase:dump IN");
    KASSERT(m_pInstanceBase);
    PyObject* pArgs = Py_BuildValue("(O)", m_pInstanceBase);
    _simple_call("dump", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:dump OUT");
  }

  void update(const std::string& _typename, const std::string& _absname)
  {
    KTRACE(m_trace, "KDIBase:update IN");
    KASSERT(m_pInstanceBase);
    PyObject* pArgs = Py_BuildValue("(O, z, z)", m_pInstanceBase, _typename.c_str(), _absname.c_str());
    KASSERT(pArgs);
    _simple_call("update", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:update OUT");
  }

  const std::string update_fields(const std::string& _nameParentMesh, const std::string& _nameField)
  {
    KTRACE(m_trace, "KDIBase:update_fields IN");
    KASSERT(m_pInstanceBase);
    PyObject* pArgs = Py_BuildValue("(O, z, z)", m_pInstanceBase, _nameParentMesh.c_str(), _nameField.c_str());
    KASSERT(pArgs);
    _simple_call("update_fields", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:update_fields OUT");
    // TODO Ceci est une verole... il serait preferable que ce soit le retour de la fonction Python qui soit exploitee
    return _nameParentMesh + "/" + _nameField;
  }

  const std::string update_sub(const std::string& _nameParentMesh, const std::string& _nameSubMesh)
  {
    KTRACE(m_trace, "KDIBase:update_sub IN");
    KASSERT(m_pInstanceBase);
    KASSERT(_nameParentMesh[0] == '/');
    KASSERT(_nameSubMesh[0] == '/');
    PyObject* pArgs = Py_BuildValue("(O, z, z)", m_pInstanceBase, _nameParentMesh.c_str(), _nameSubMesh.c_str());
    KASSERT(pArgs);
    _simple_call("update_sub", pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:update_sub OUT");
    // TODO Ceci est une verole... il serait preferable que ce soit le retour de la fonction Python qui soit exploitee
    return _nameParentMesh + "/submeshes" + _nameSubMesh;
  }

 private:

  KDIChunk* _chunk(PyObject* pArgs)
  {
    KTRACE(m_trace, "KDIBase:_chunk IN");
    PyObject* pResult = _simple_call("chunk", pArgs);
    Py_DECREF(pArgs);
    KDIChunk* chunk = new KDIChunk(pResult, m_trace);
    KTRACE(m_trace, "KDIBase:_chunk OUT");
    return chunk;
  }

 public:

  KDIChunk* chunk()
  {
    KTRACE(m_trace, "KDIBase:chunk IN");
    KASSERT(m_pInstanceBase);
    PyObject* pArgs = Py_BuildValue("(O)", m_pInstanceBase);
    KDIChunk* chunk = _chunk(pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:chunk OUT (" << chunk << ")");
    return chunk;
  }

  KDIChunk* chunk(double _vstep)
  {
    KTRACE(m_trace, "KDIBase:chunk (" << _vstep << ") IN");
    KASSERT(m_pInstanceBase);
    PyObject* pArgs = Py_BuildValue("(O, f)", m_pInstanceBase, _vstep);
    KDIChunk* chunk = _chunk(pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:chunk OUT (" << chunk << ")");
    return chunk;
  }

  KDIChunk* chunk(double _vstep, int _ipart)
  {
    KTRACE(m_trace, "KDIBase:chunk (" << _vstep << ") IN");
    KASSERT(m_pInstanceBase);
    PyObject* pArgs = Py_BuildValue("(O, f, i)", m_pInstanceBase, _vstep, _ipart);
    KDIChunk* chunk = _chunk(pArgs);
    Py_DECREF(pArgs);
    KTRACE(m_trace, "KDIBase:chunk OUT (" << chunk << ")");
    return chunk;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KDIBase*
createBase(unsigned int _nb_parts, bool _trace = false)
{
  KTRACE(_trace, "KDI C++ createBase (" << _nb_parts << ") IN");
  PyObject* pValue = PyUnicode_FromString("pykdi");
  KASSERT(pValue);
  KTRACE(_trace, "KDI   FromString pValue=" << pValue);
  KTRACE(_trace, "KDI   Getenv PYTHONPATH=" << getenv("PYTHONPATH"));
  KTRACE(_trace, "KDI   Getenv KDI_DICTIONARY_PATH=" << getenv("KDI_DICTIONARY_PATH"));
  // Si on a un plantage, c'est parce que les commandes :
  //      Py_Initialize();
  //      import_array();
  // n'ont pas été exécutées avant.
  PyObject* pModule = PyImport_Import(pValue);
  KTRACE(_trace, "KDI   Import ?");
  KTRACE(_trace, "KDI   pModule=" << pModule);
  KASSERT(pModule);
  KTRACE(_trace, "KDI   Import");
  PyObject* pClass = PyObject_GetAttrString(pModule, "KDIAgreementStepPartBase");
  KASSERT(pClass);
  KTRACE(_trace, "KDI   GetAttrString");
  KASSERT(PyCallable_Check(pClass));
  KTRACE(_trace, "KDI   Check");
  PyObject* pArgs = Py_BuildValue("(i)", (int)_nb_parts);
  KASSERT(pArgs);
  KTRACE(_trace, "KDI   BuildValue");
  PyObject* pResult = PyObject_CallObject(pClass, pArgs);
  KASSERT(pResult);
  KTRACE(_trace, "KDI   CallObject");
  Py_DECREF(pArgs);
  Py_DECREF(pClass);
  Py_DECREF(pModule);
  Py_DECREF(pValue);
  KDIBase* base = new KDIBase(pResult, _trace);
  KTRACE(_trace, "KDI C++ createBase OUT");
  return base;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KDIBase*
loadVTKHDF(const std::string _absfilename, bool _trace = false)
{
  KTRACE(_trace, "loadBase () IN");
  PyObject* pValue = PyUnicode_FromString("pykdi");
  KASSERT(pValue);
  PyObject* pModule = PyImport_Import(pValue);
  KASSERT(pModule);
  PyObject* pClass = PyObject_GetAttrString(pModule, "KDIAgreementStepPartBase");
  KASSERT(pClass);
  KASSERT(PyCallable_Check(pClass));
  PyObject* pArgs = Py_BuildValue("(z)", _absfilename.c_str());
  KASSERT(pArgs);
  PyObject* pResult = PyObject_CallObject(pClass, pArgs);
  KASSERT(pResult);
  Py_DECREF(pArgs);
  Py_DECREF(pClass);
  Py_DECREF(pModule);
  Py_DECREF(pValue);
  KDIBase* base = new KDIBase(pResult, _trace);
  KTRACE(_trace, "loadVTKHDF OUT");
  return base;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Convert a c++ 2D vector into a numpy array
 *
 * @param const vector< vector<T> >& vec : 2D vector data
 * @return PyArrayObject* array : converted numpy array
 *
 * Transforms an arbitrary 2D C++ vector into a numpy array. Throws in case of
 * unregular shape. The array may contain empty columns or something else, as
 * long as it's shape is square.
 *
 * Warning this routine makes a copy of the memory!
 */
template <typename T>
static PyArrayObject* vector_to_nparray(const std::vector<std::vector<T>>& vec, int type_num = NPY_FLOAT64)
{

  // rows not empty
  if (!vec.empty()) {

    // column not empty
    if (!vec[0].empty()) {

      size_t nRows = vec.size();
      size_t nCols = vec[0].size();
      npy_intp dims[2] = { static_cast<npy_intp>(nRows), static_cast<npy_intp>(nCols) };
      PyArrayObject* vec_array = (PyArrayObject*)PyArray_SimpleNew(2, dims, type_num);

      T* vec_array_pointer = (T*)PyArray_DATA(vec_array);

      // copy vector line by line ... maybe could be done at one
      for (size_t iRow = 0; iRow < vec.size(); ++iRow) {

        if (vec[iRow].size() != nCols) {
          Py_DECREF(vec_array); // delete
          throw(std::string("Can not convert vector<vector<T>> to np.array, since c++ matrix shape is not uniform."));
        }

        copy(vec[iRow].begin(), vec[iRow].end(), vec_array_pointer + iRow * nCols);
      }

      return vec_array;

      // Empty columns
    }
    else {
      npy_intp dims[2] = { vec.size(), 0 };
      return (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    }

    // no data at all
  }
  else {
    npy_intp dims[2] = { 0, 0 };
    return (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
  }
}

/** Convert a c++ vector into a numpy array
 *
 * @param const vector<T>& vec : 1D vector data
 * @return PyArrayObject* array : converted numpy array
 *
 * Transforms an arbitrary C++ vector into a numpy array. Throws in case of
 * unregular shape. The array may contain empty columns or something else, as
 * long as it's shape is square.
 *
 * Warning this routine makes a copy of the memory!
 *
 *
 *
 * https://pyo3.github.io/rust-numpy/numpy/npyffi/types/enum.NPY_TYPES.html
 */
template <typename T>
static PyArrayObject* vector_to_nparray(const std::vector<T>& vec, int type_num = NPY_FLOAT64, int comp_num = 1)
{
  bool trace{ true };

  // rows not empty
  if (!vec.empty()) {
    PyArrayObject* vec_array = nullptr;
    if (comp_num == 1) {
      int nd = 1;
      npy_intp dims[]{ vec.size() };
      vec_array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, type_num);
    }
    else {
      int nd = 2;
      npy_intp dims[]{ int(vec.size() / comp_num), comp_num };
      vec_array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, type_num);
    }
    T* vec_array_pointer = (T*)PyArray_DATA(vec_array);
    KTRACE(trace, vec_array_pointer);
    // A bannir
    copy(vec.begin(), vec.end(), vec_array_pointer);
    return vec_array;

    // no data at all
  }
  else {
    npy_intp dims[1] = { 0 };
    return (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
