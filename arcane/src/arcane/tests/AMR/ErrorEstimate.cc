// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ErrorEstimate.cc                                            (C) 2000-2010 */
/*                                                                           */
/* Service de solutions analytiques utilisees pour estimer l'erreur AMR.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#ifndef __ERROR_ESTIMATE_h__
#define __ERROR_ESTIMATE_h__


using namespace Arcane;

class ErrorEstimate
{

	enum NormType {L2              = 0,
		       L_INF           = 1,
		       INVALID_NORM    = 10};

public:
	/**
	 * Constructor.
	 */
	ErrorEstimate ();
	//ErrorEstimate(const ServiceBuildInfo & sbi);
	/**
	 * Destructor.
	 */
	~ErrorEstimate() {}
	/**
	 * Attacher une fonction arbitraire calculant
	 * la valeur exacte de la solution dans un point donné.
	 */
	void attachExactValue ( Real fptr(const Real3& p));

	/**
	 * Attacher une fonction arbitraire calculant
	 * la valeur exacte du gradient de la solution dans un point donné.
	 */
	void attachExactGradient ( Real3 fptr(const Real3& p));
	/**
	 * Attacher une fonction arbitraire calculant
	 * la valeur exacte du hessien de la solution dans un point donné.
	 */
	void attachExactHessian ( Real3x3 fptr(const Real3& p));
	/**
	 * Calcul et stockage de l'erreur de la solution e = u-u_h,
	 * du gradient grad(e) = grad(u) - grad(u_h), et aussi le hessien
	 * grad(grad(e)) = grad(grad(u)) - grad(grad(u_h)).
	 */
	void computeSol(RealArray & sol,IMesh* mesh);

	void computeGlobalError();
	void computeError(RealArray& error,IMesh* mesh);
	void errorToFlagConverter(RealArray& error_per_cell, const Real& refine_frac,
		      const Real& coarsen_frac,const Integer& max_level,IMesh* mesh);
	/**
	 * erreur L2.
	 * Note: pas de calcul de l'erreur,
	 * il faut appeler d'abord le compute_error()
	 * .
	 */
	Real l2Error();

	/**
	 * erreur LInf.
	 * Note: pas de calcul de l'erreur,
	 * il faut appeler d'abord le compute_error()
	 * .
	 */
	Real lInfError();
	/**
	 * Cette methode retourne l'erreur dans la norme demandée.
	 *  Note: pas de calcul de l'erreur,
	 * il faut appeler d'abord le compute_error()
	 */
	Real errorNorm(const NormType& norm);
private:

	/**
	 * Function pointer à une fonction fournit par l'utilisateur
	 * Celle-ci calcule la valeur exacte de la solution.
	 */
	Real (* m_exact_value) (const Real3& p);
	/**
		 * Function pointer à une fonction fournit par l'utilisateur
		 * Celle-ci calcule la dérivée exacte de la solution.
		 */

	  Real3 (* m_exact_gradient) (const Real3& p);
	  /**
	  	 * Function pointer à une fonction fournit par l'utilisateur
	  	 * Celle-ci calcule la dérivéee seconde exacte de la solution.
	  	 */
	  Real3x3 (* m_exact_hessian)(const Real3& p);
	/**
	 * Calcul l'erreur sur la solution et ses dérivées pour un système scalaire.
	 * elle peut être utilisée pour résoudre des systèmes vectoriels
	 */
	void _computeError(RealArray& error_vals,IMesh* mesh);

	/**
	 * Vecteur propre au stockage de l'erreur globale.
	 */
	Real3 m_error_vals;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ErrorEstimate::ErrorEstimate() :
  m_exact_value (NULL),
  m_exact_gradient(NULL)
{

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::attachExactValue (Real fptr(const Real3& p))
{
  ARCANE_ASSERT ((fptr != NULL),(""));
  m_exact_value = fptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::attachExactGradient (Real3 fptr(const Real3& p))
{
  ARCANE_ASSERT ((fptr != NULL),(""));
  m_exact_gradient = fptr;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::attachExactHessian (Real3x3 fptr(const Real3& p))
{
  ARCANE_ASSERT ((fptr != NULL),(""));
  m_exact_hessian = fptr;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
computeSol(RealArray & sol,IMesh* mesh)
{
  //
  SharedVariableNodeReal3 orig_nodes_coords(mesh->sharedNodesCoordinates());
  // sol.reserve(mesh->allActiveCells().size());

  ENUMERATE_CELL(icell,mesh->allCells()){ // active_local_elements
	  const Cell& cell = *icell;
	  const Int32 nb_nodes = cell.nbNode();

	  Real3 cellCenter(Real3::null());
	  for( Integer i=0; i<nb_nodes; ++i )
		  cellCenter += orig_nodes_coords[cell.node(i)];
	  cellCenter /= nb_nodes;

	  // calcul de l'erreur au centre de la maille
	  Real exact_val = 0.0;
    if (m_exact_value){
	    //exact_val = m_exact_value(cellCenter);
	    exact_val = m_exact_value(cellCenter);
	    sol.add(exact_val);
    }
  } // end ownActiveCell enumerate
  std::cout << "SOL_SIZE=" << sol.size() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::computeError(RealArray& error,IMesh* mesh)
{
  this->_computeError(error,mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ErrorEstimate::errorNorm(const NormType& norm)
{

  switch (norm)
    {
    case L2:
      return m_error_vals[0];
    case L_INF:
      return m_error_vals[1];
    // \todo à étendre pour d'autres normes/semi-normes
    default:
    	throw FatalErrorException(A_FUNCINFO,String::format("Norm Type is not implemented!"));
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ErrorEstimate::l2Error()
{
  // Return la norme L2 de l'erreur.
  return m_error_vals[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ErrorEstimate::lInfError()
{

  // Return la norme inf de l'erreur.
  return m_error_vals[1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ErrorEstimate::
_computeError(RealArray & error,IMesh* mesh)
{
  // Vérification de la solution et ses dérivées
  //ARCANE_ASSERT ((!(m_exact_value && m_exact_gradient)), (""));


  // initialisation à Zero de l'erreur avant sommation
  m_error_vals = Real3::null();


  // Les tests sont stationnaires
  // const Real time = 0.;//

  //
  SharedVariableNodeReal3 orig_nodes_coords(mesh->sharedNodesCoordinates());
  error.reserve(mesh->allActiveCells().size());
  error.fill(0.);
  ENUMERATE_CELL(icell,mesh->allCells()){ // active_local_elements
	  Cell cell = *icell;
	  const Int32 nb_nodes = cell.nbNode();

	  Real3 cellCenter(Real3::null());
	  for( Integer i=0; i<nb_nodes; ++i )
		  cellCenter += orig_nodes_coords[cell.node(i)];
	  cellCenter /= nb_nodes;

	  // calcul de l'erreur au centre de la maille
	  Real3 grad_exact_value(Real3::null());
    if (m_exact_gradient){
	    //exact_val = m_exact_value(cellCenter);
	    grad_exact_value = m_exact_gradient(cellCenter);
    }

    const Real& val_error = pow(grad_exact_value[0],2) + pow(grad_exact_value[1],2) +
    pow(grad_exact_value[2],2);// x**2 +y**2+z**2

    error.add(math::sqrt(val_error));
    // Assemblage de l'erreur
    m_error_vals[0] += val_error;
    Real norm = math::sqrt(val_error);

    if(m_error_vals[1]<norm)
    { m_error_vals[1] = norm; }


  } // end ownActiveCell enumerate

  // Add up the error values on all processors, except for the L-infty
  // norm, for which the maximum is computed.
  IParallelMng* pm = mesh->parallelMng();

  if (pm->commSize() > 1){
    const Real& l2 =  m_error_vals[0];
    const Real& l_inf = m_error_vals[1];
    Real l_inf_g, l2_g;
    l_inf_g = pm->reduce(Parallel::ReduceMax,l_inf);
    l2_g = pm->reduce(Parallel::ReduceSum,l2);
    m_error_vals[0] = math::sqrt(l2_g);
    m_error_vals[1] = l_inf_g;
  }
  else
    m_error_vals[0]=math::sqrt(m_error_vals[0]);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// passage de l'erreur commise par maille au flag de raffinement
// Cette méthode pourrait être conçue de manières différentes:
// 1- implémentation actuelle: l'uilisateur fait la transformation lui-même
// dans ce cas, il modifie l'objet itemInternal en settant le flag de raffinement
// 2- l'uilisateur fait la transformation lui-même mais stocke et retourne un tableau des flags
// la classe MeshRefinement, dans ce cas là, implémente un setter à partir du tableau retourné ici
// 3- pour éviter la copie du tableau des flags, implémenter le converter directement dans meshRefinement
// et l'utilisateur ne fait que fournir le tableau d'erreur
void ErrorEstimate::
errorToFlagConverter(RealArray& error_per_cell, const Real& refine_frac,
                     const Real& coarsen_frac,const Integer& max_level,IMesh* mesh)
{

  // Check for valid fractions..
  // The fraction values must be in [0,1]
  ARCANE_ASSERT ((refine_frac  >= 0. && refine_frac  <= 1.),(" 0 <= refine_frac  <= 1."));
  ARCANE_ASSERT ((coarsen_frac >= 0. && coarsen_frac <= 1.),("0 <= coarsen_frac <= 1."));


  // We're getting the minimum and maximum error values
  // for the ACTIVE elements
  Real error_min = 1.e30;
  Real error_max = 0.;

  // We need to loop over all active elements to find the minimum
  IParallelMng* pm = mesh->parallelMng();
  Integer i=0;
  ENUMERATE_CELL(icell,mesh->ownActiveCells()){ // active cells
    Cell cell = *icell;
    const Integer id  = cell.localId();
    if (id >= error_per_cell.size())
      ARCANE_FATAL("Bad local_id '{0}' (max_valid={1})",id,error_per_cell.size());

    error_max = math::max (error_max, error_per_cell[i]);
    error_min = math::min (error_min, error_per_cell[i]);
    i++;
  }
  if(pm->commSize()>1){
    const Real error_max_g= pm->reduce(Parallel::ReduceMax,error_max);
    const Real error_min_g= pm->reduce(Parallel::ReduceMin,error_min);
    error_max= error_max_g;
    error_min= error_min_g;
  }


  // Compute the cutoff values for coarsening and refinement
  const Real error_delta = (error_max - error_min);
  //const Real parent_error_delta = parent_error_max - parent_error_min;

  const Real refine_cutoff  = (1.- refine_frac)*error_max;
  const Real coarsen_cutoff = coarsen_frac*error_delta + error_min;

//   // Print information about the error
//   debug() << " Error Information: \n"                     <<
// 	    << " ------------------\n"                     <<
// 	    << "   min:              " << error_min      << "\n"
// 	    << "   max:              " << error_max      << "\n"
// 	    << "   delta:            " << error_delta    << "\n"
// 	    << "     refine_cutoff:  " << refine_cutoff  << "\n"
// 	    << "     coarsen_cutoff: " << coarsen_cutoff << "\n";



  // Tag les mailles pour adaptation
  i=0;
  ENUMERATE_CELL(icell,mesh->ownActiveCells()){ // active cells
	  Cell cell = *icell;
	  ItemInternal* iitem = cell.internal();
	  const Integer id  = cell.localId();
	  ARCANE_ASSERT ((id < error_per_cell.size()),("cell_lid < error_per_cell.size()"));

	  const Real cell_error = error_per_cell[i++];

	  // Flag pour deraffinement si error <= coarsen_fraction*delta + error_min
	  if (cell_error <= coarsen_cutoff && cell.level() > 0)
	  {
		  if (cell.type()==IT_Hexaedron8){
			  Integer f = iitem->flags();
			  f |= ItemInternal::II_Coarsen;
			  iitem->setFlags(f);
		  }

	  }

	  // Flag pour raffinement  si error >= refinement_cutoff.
	  if (cell_error >= refine_cutoff && cell.level() < max_level)
			  if (cell.type()==IT_Hexaedron8){
				  Integer f = iitem->flags();
				  f |= ItemInternal::II_Refine;
				  iitem->setFlags(f);
			  }
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif
