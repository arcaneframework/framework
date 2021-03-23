// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "arcane/IMesh.h"

#include "Utils/Utils.h"

#include "Numerics/BCs/BoundaryConditionTypes.h"
#include "Numerics/BCs/IBoundaryCondition.h"
#include "Numerics/BCs/IBoundaryConditionMng.h"

#include "Numerics/BCs/ScalarBoundaryCondition.h"

#include "Numerics/BCs/BoundaryConditionMngService.h"

#include "Appli/IAppServiceMng.h"
#include "ExpressionParser/IExpressionParser.h"
#include "Utils/ItemGroupBuilder.h"

using namespace Arcane;

void BoundaryConditionMngService::init() {
  if(m_initialized) return;

  // Retrieve application service manager to access shared services
  IAppServiceMng* app_service_mng = IAppServiceMng::instance(subDomain()->serviceMng());

  // Retrieve shared expression parser service
  IExpressionParser* ep = app_service_mng->find<IExpressionParser>(true);

  // Add boundary conditions
  m_bcs.resize( options()->bc.size() );

  for(int i = 0; i < options()->bc.size(); i++) {
    ep->parse(options()->bc[i]->value());
    BoundaryConditionPtr bc_ptr =
      new ScalarBoundaryCondition( options()->bc[i]->type(), ep->getResult() );

    m_bcs[i]                                      = bc_ptr;
    m_tag_2_bc[ "BC_" + options()->bc[i]->tag() ] = bc_ptr;
  }

  // Check that corresponding face groups exist and build
  // face unique id to boundary condition map
  for(Tag2BoundaryConditionPtrMap::iterator ibc = m_tag_2_bc.begin(); ibc != m_tag_2_bc.end(); ibc++) {
    const ItemGroup& bc_face_group = subDomain()->mesh()->findGroup( ibc->first );

    if( bc_face_group.null() ){
      
      if(ibc->first == "BC_allBoundaryFaces"){
        ENUMERATE_FACE( iface, subDomain()->mesh()->allFaces() ) {
          const Face& F  = *iface;
          if( F.isBoundary() ) {
            m_face_2_bc[F] = ibc->second;
          }
        }
      }
      else	
      warning() << "[BoundaryConditionMngService::init] Face group "
                << ibc->first << " does not exist";
    
    }
    else {
      ENUMERATE_FACE( iface, bc_face_group ) {
        const Face& F  = *iface;
        m_face_2_bc[F] = ibc->second;
      }
    }
  }

  m_initialized = true;

  info() << "[BoundaryConditionMngService::init]"
         << " Boundary condition manager initialized";
}

////////////////////////////////////////////////////////////

BoundaryConditionMngService::BoundaryConditionPtr
BoundaryConditionMngService::getBoundaryCondition(const Face& F) {
  Face2BoundaryConditionPtrMap::iterator bc_it = m_face_2_bc.find(F);

  if( bc_it == m_face_2_bc.end() )
    error() << "No boundary condition associated with face " << F.uniqueId();

  return bc_it->second;
}

FaceGroup BoundaryConditionMngService::getFacesOfType(BoundaryConditionTypes::eType type) {
  ItemGroupBuilder<Face> faces_builder(this->subDomain()->mesh(), IMPLICIT_NAME);
  for(Face2BoundaryConditionPtrMap::iterator i = m_face_2_bc.begin(); i != m_face_2_bc.end(); i++)
    if(i->second->getType() == type) faces_builder.add(i->first);
  return faces_builder.buildGroup();
}

FaceGroup BoundaryConditionMngService::getFacesNotOfType(BoundaryConditionTypes::eType type) {
  ItemGroupBuilder<Face> faces_builder(this->subDomain()->mesh(), IMPLICIT_NAME);
  for(Face2BoundaryConditionPtrMap::iterator i = m_face_2_bc.begin(); i != m_face_2_bc.end(); i++)
    if(i->second->getType() != type) faces_builder.add(i->first);
  return faces_builder.buildGroup();
}

ARCANE_REGISTER_SERVICE_BOUNDARYCONDITIONMNG(BoundaryConditionMng,
                                             BoundaryConditionMngService);

