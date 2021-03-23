// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_TESTS_SERVICETESTERS_TOYREACTIVETRANSPORTMODELTESTER_H
#define ARCGEOSIM_TESTS_SERVICETESTERS_TOYREACTIVETRANSPORTMODELTESTER_H
/* Author : gratienj at Fri Aug 29 15:45:29 2008
 * 
 */


#include "Tests/ServiceTesters/IServiceTester.h"

using namespace Arcane;

#include "ToyReactiveTransportModelTester_axl.h"

class IGeometryMng;
class ITimeMng ;
class ITimeStepMng ;

class ToyReactiveTransportModelTester 
: public ArcaneToyReactiveTransportModelTesterObject
{
 public:
  /** Constructeur de la classe */
   ToyReactiveTransportModelTester(const Arcane::ServiceBuildInfo & sbi) 
   : ArcaneToyReactiveTransportModelTesterObject(sbi)
   , m_time_mng(NULL)
   , m_model(NULL)
   , m_collector(NULL)
   , m_validator(NULL)
   {
      ;
   }
  
  /** Destructeur de la classe */
  virtual ~ToyReactiveTransportModelTester() 
  {
  }
  
 public:

  //! Initialization
  void init();
  //! Run the test
  int test();
  
  bool goOn() ;

 private:
   typedef enum {
   DtSeq, //! time step computation sequence
   } eTimeStepSeqType ;
   
   void initModelDomains() ;
   void setBoundaryCondition(ISubDomainModel* model,
                           Array<FaceGroup>& bc_groups) ;
   void readGroup();
   void computeInitialState() ;
   void computeDomain() ;
   void computeDeltaT() ;
   
   //!copy of cell collections, warning group and collection should be compatible
   template<class Collection1, class Collection2>
   void _copy(const CellGroup& group,Collection1& pressure1,const Collection2& pressure2)
   {
     ENUMERATE_CELL(icell,group)
     {
       pressure1[icell] = pressure2[icell] ;
     }
   }
   //!safe copy of cell collection, group and collection can be not compatible
   template<class Collection1, class Collection2>
   void _copy2(const CellGroup& group,Collection1& pressure1,const Collection2& pressure2)
   {
     ENUMERATE_CELL(icell,group)
     {
       const Cell& cell = *icell ;
       pressure1[cell] = pressure2[cell] ;
     }
   }
   //!save and copy of cell collections, warning group and collection should be compatible
   template<class Collection1, class Collection2, class Collection3>
   void _copysave(const CellGroup& group,
                  Collection1& pressure1,
                  Collection2& pressure2,
                  const Collection3& pressure3)
   {
     ENUMERATE_CELL(icell,group)
     {
       pressure1[icell] = pressure2[icell] ;
       pressure2[icell] = pressure3[icell] ;
     }
   }
   //!safe save and copy of cell collections, group and collection can be not compatible
   template<class Collection1, class Collection2, class Collection3>
   void _copysave2(const CellGroup& group,
                  Collection1& pressure1,
                  Collection2& pressure2,
                  const Collection3& pressure3)
   {
     ENUMERATE_CELL(icell,group)
     {
       const Cell& cell = *icell ;
       pressure1[cell] = pressure2[cell] ;
       pressure2[cell] = pressure3[cell] ;
     }
   }
   
   ITimeMng* m_time_mng ;
   ITimeStepMng* m_time_step_mng ;
  
   INumericalModel *m_model ;
   
   ICollector* m_collector ;
   
   //!Reservoir domain description
   //!@{
   FaceGroup m_internal_face;
   FaceGroup m_boundary_face;
   //!@}
  
   Array<FaceGroup> m_bc_group ;
   
   Integer m_error ;
   
   IServiceValidator* m_validator ;
   
   Integer m_output_level ;

};

#endif /* ARCGEOSIM_TESTS_SERVICETESTERS_TOYREACTIVETRANSPORTMODELTESTER_H */
