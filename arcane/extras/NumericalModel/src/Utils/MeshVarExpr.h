// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef MESHVAREXPR_H_
#define MESHVAREXPR_H_

class ScalarMeshVar
{
public :
  ScalarMeshVar(Real a)
  : m_a(a)
  {}
  Real operator[](const ItemEnumeratorT<Cell>& icell) const
  {
    return m_a ;
  }
  Real operator[](const ItemGroupRangeIteratorT<Cell>& icell) const
  {
    return m_a ;
  }
  Real operator[](const Cell& cell) const
  {
    return m_a ;
  }
private :
  Real m_a;
} ;

template<class L,class R>
class MeshVarBinaryAdd ;

template<class R>
class MeshVarBinaryAdd<Real,R>
{
public :
  static Real eval(const Real& l,const R& r,const ItemEnumeratorT<Cell>& icell)
  {
    return l + r[icell] ; 
  }
  static Real eval(const Real& l,const R& r,const ItemGroupRangeIteratorT<Cell>& icell)
  {
    return l + r[icell] ; 
  }
  static Real eval(const Real& l,const R& r,const Cell& cell)
  {
    return l + r[cell] ;
  }
} ;

template<class L>
class MeshVarBinaryAdd<L,Real>
{
public :
  static Real eval(const L& l,const Real& r,const ItemEnumeratorT<Cell>& icell)
  { return l[icell] + r ; }
  
   static Real eval(const L& l,const Real& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l[icell] + r ; }
   
   static Real eval(const L& l,const Real& r,const Cell& cell)
   { return l[cell] + r ; }
} ;

template<class L,class R>
class MeshVarBinaryAdd
{
public :
  static Real eval(const L& l,const R& r,const ItemEnumeratorT<Cell>& icell)
  { return l[icell] + r[icell] ; }

   static Real eval(const L& l,const R& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l[icell] + r[icell] ; }
 
   static Real eval(const L& l,const R& r,const Cell& cell)
   { return l[cell] + r[cell] ; }
} ;

template<class L,class R>
class MeshVarBinaryMult ;

template<class R>
class MeshVarBinaryMult<Real,R>
{
public :
  static Real eval(const Real& l,const R& r,const ItemEnumeratorT<Cell>& icell)
  { return l * r[icell] ; }
  
   static Real eval(const Real& l,const R& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l * r[icell] ; }
   
   static Real eval(const Real& l,const R& r,const Cell& cell)
   { return l * r[cell] ; }
} ;

template<class L>
class MeshVarBinaryMult<L,Real>
{
public :
  static Real eval(const L& l,const Real& r,const ItemEnumeratorT<Cell>& icell)
  { return l[icell] * r ; }

   static Real eval(const L& l,const Real& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l[icell] * r ; }
 
   static Real eval(const L& l,const Real& r,const Cell& cell)
   { return l[cell] * r ; }
} ;

template<class L,class R>
class MeshVarBinaryMult
{
public :
  static Real eval(const L& l,const R& r,const ItemEnumeratorT<Cell>& icell)
  { return l[icell] * r[icell] ; }
  
   static Real eval(const L& l,const R& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l[icell] * r[icell] ; }
   
   static Real eval(const L& l,const R& r,const Cell& cell)
   { return l[cell] * r[cell] ; }
} ;


template<class L,class R>
class MeshVarBinaryDiv ;

template<class R>
class MeshVarBinaryDiv<Real,R>
{
public :
  static Real eval(const Real& l,const R& r,const ItemEnumeratorT<Cell>& icell)
  { return l / r[icell] ; }
  
   static Real eval(const Real& l,const R& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l / r[icell] ; }
   
   static Real eval(const Real& l,const R& r,const Cell& cell)
   { return l / r[cell] ; }
} ;

template<class L>
class MeshVarBinaryDiv<L,Real>
{
public :
  static Real eval(const L& l,const Real& r,const ItemEnumeratorT<Cell>& icell)
  { return l[icell] / r ; }

   static Real eval(const L& l,const Real& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l[icell] / r ; }
 
   static Real eval(const L& l,const Real& r,const Cell& cell)
   { return l[cell] / r ; }
} ;

template<class L,class R>
class MeshVarBinaryDiv
{
public :
  static Real eval(const L& l,const R& r,const ItemEnumeratorT<Cell>& icell)
  { return l[icell] / r[icell] ; }
  
   static Real eval(const L& l,const R& r,const ItemGroupRangeIteratorT<Cell>& icell)
   { return l[icell] / r[icell] ; }
   
   static Real eval(const L& l,const R& r,const Cell& cell)
   { return l[cell] / r[cell] ; }
} ;
template<class L,class Op,class R>
class MeshVarExpr
{
public :
  MeshVarExpr(const L& l,const R& r)
  : m_l(l)
  , m_r(r)
  {}
  Real operator[](const ItemEnumeratorT<Cell>& icell) const
  {
    return Op::eval(m_l,m_r,icell) ;
  }
  Real operator[](const ItemGroupRangeIteratorT<Cell>& icell) const
  {
    return Op::eval(m_l,m_r,icell) ;
  }
  Real operator[](const Cell& cell) const
  {
    return Op::eval(m_l,m_r,cell) ;
  }
private :
  const L& m_l ;
  const R& m_r ;
} ;

namespace MeshVariableOperator
{
  template<class L,class R>
  inline MeshVarExpr<L,MeshVarBinaryAdd<L,R>,R> operator+(const L& l, const R& r)
  {
    return MeshVarExpr<L,MeshVarBinaryAdd<L,R>,R>(l,r) ;
  }
  
  template<class L,class R>
  inline MeshVarExpr<L,MeshVarBinaryMult<L,R>,R> operator*(const L& l, const R& r)
  {
    return MeshVarExpr<L,MeshVarBinaryMult<L,R>,R>(l,r) ;
  }
  
  template<class L,class R>
  inline MeshVarExpr<L,MeshVarBinaryDiv<L,R>,R> operator/(const L& l, const R& r)
  {
    return MeshVarExpr<L,MeshVarBinaryDiv<L,R>,R>(l,r) ;
  }
}

#endif /*MESHVAREXPR_H_*/
