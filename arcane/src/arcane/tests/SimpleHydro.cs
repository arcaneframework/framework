using System;
using Arcane;
using Real = System.Double;
#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Math = Arcane.Math;
using System.Threading.Tasks;
using System.Collections.Generic;

public class TypesSimpleHydro
{
  public enum eViscosity
    {
      ViscosityNo,
      ViscosityCellScalar
    }
  public enum eBoundaryCondition
    {
      VelocityX, //!< Vitesse X fixée
      VelocityY, //!< Vitesse Y fixée
      VelocityZ, //!< Vitesse Z fixée
      Unknown    //!< Type inconnu
    };
};

public interface IEquationOfState
{
  void initEOS(CellGroup group);
  void applyEOS(CellGroup group);
}

[Arcane.Module("SimpleHydroCS","1.2.3")]
class SimpleHydroModule : ArcaneSimpleHydroCSObject
{

  private IMesh m_mesh;

  Real m_density_ratio_maximum; //!< Accroissement maximum de la densité sur un pas de temps
  Real m_delta_t_n; //!< Delta t n entre t^{n-1/2} et t^{n+1/2}
  Real m_delta_t_f; //!< Delta t n+\demi  entre t^{n} et t^{n+1}
  Real m_old_dt_f; //!< Delta t n-\demi  entre t^{n-1} et t^{n}

  public SimpleHydroModule(ModuleBuildInfo infos) : base(infos)
  {
    Console.WriteLine("SimpleHydroModule C#!");
  }

  public override void HydroStartInit()
  {
    m_mesh = DefaultMesh();
    // Dimensionne les variables tableaux
    m_cell_cqs.Resize(8);

    m_global_deltat.Value = Options.DeltatInit;

    Trace.Info("VAR Name v={0}",m_density.Name());
    Trace.Info("VAR Size v={0}",m_density.AsArray().Length);

    CellGroup g = m_mesh.FindGroup("ZG").CellGroup();
    Trace.Info("FIND GROUP ZG");
    Trace.Info("END FIND GROUP ZG {0}",g.Size());
    {
      foreach( Cell cell in g ){
        //Trace.Info("ID1 = {0} {1}",item.LocalId,item.UniqueId);
        //Cell cell = item.ToCell();
        m_density[cell] = 1.0;
        m_pressure[cell] = 1.0;
        m_adiabatic_cst[cell] = 1.4;
      }
    }
    
    Trace.Info("FIND GROUP ZD");
    ItemGroup g2 = m_mesh.FindGroup("ZD");
    if (g2.IsNull())
      throw new ArgumentException("Can not find group 'ZD'");
    Trace.Info("END FIND GROUP ZD {0}",g2.Size());
    {
      int index = 0;
      foreach( Item item in g2 ){
        //Trace.Info("INDEX = {0}",index);
        //Trace.Info("ID2 = {0} {1}",item.LocalId,item.UniqueId);
        Cell cell = item.ToCell();
        m_density[cell] = 0.125;
        m_pressure[cell] = 0.1;
        m_adiabatic_cst[cell] = 1.4;
        ++index;
      }
    }
    //int[] toto = new int[10];
    //Parallel.ForEach(toto,(int x) => { Trace.Info("I={0}"); });

    // Initialise le delta-t
    Real deltat_init = Options.DeltatInit;
    m_delta_t_n = deltat_init;
    m_delta_t_f = deltat_init;

    Trace.Info("COMPUTE GEOM");
    // Initialise les données géométriques: volume, cqs, longueurs caractéristiques
    ComputeGeometricValues();
    
    Trace.Info("FILL");
    m_node_mass.Fill(0.0);
    m_velocity.Fill(Real3.Zero);

    Trace.Info("COMPUTE NODE MASS");
    // Initialisation de la masse des mailles et des masses nodale
    foreach(Cell icell in AllCells()){
      m_cell_mass[icell] = m_density[icell] * m_volume[icell];

      Real contrib_node_mass = 0.125 * m_cell_mass[icell];
      foreach(Node node in icell.Nodes){ // int i=0; i<icell.NbNode; ++i ){
        //for( int i=0; i<icell.NbNode; ++i ){
        //m_node_mass[icell.Node(i)] += contrib_node_mass;
        m_node_mass[node] += contrib_node_mass;
      }
    }

    m_node_mass.Synchronize();

    Trace.Info("INIT EOS");
    ITraceMng tm = SubDomain().TraceMng();
    tm.PutTrace("Hello",1);

    Trace.Info("Hello");
    // Initialise l'énergie et la vitesse du son
    Options.EosModel.initEOS(AllCells());
    Trace.Info("NB_EOS_ARRAY={0}",Options.EosModelArray.Length);
    foreach(IEquationOfState eos in Options.EosModelArray){
      Trace.Info("EOS_ARRAY {0}",eos);
    }
  }

  public override void HydroContinueInit()
  {
  }

  public override void ComputePressureForce()
  {
    // Remise à zéro du vecteur des forces.
    //Real3 null_real3 = new Real3(0.0,0.0,0.0);
    //m_force.Fill(null_real3);
    m_force.Fill(Real3.Zero);
    //Trace.Info("COMPUTE PRESSURE FORCE 2");
    // Calcul pour chaque noeud de chaque maille la contribution
    // des forces de pression
    foreach(Cell icell in AllCells()){
      Real pressure = m_pressure[icell];
      foreach(IndexedNode node in icell.Nodes){ // int i=0; i<icell.NbNode; ++i ){
        //Trace.Info("ADD ME node={0}",icell.Node(i));
        //m_force[icell.Node(i)] += pressure * m_cell_cqs[icell][i];
        m_force[node] += pressure * m_cell_cqs[icell][node.Index];
      }
    }
    //Trace.Info("END COMPUTE PRESSURE FORCE");
    m_force.Synchronize();
  }

  public override void ComputePseudoViscosity()
  {
    if (Options.Viscosity!=TypesSimpleHydro.eViscosity.ViscosityNo)
      cellScalarPseudoViscosity();
  }

  void cellScalarPseudoViscosity()
  {
    //Trace.Info("COMPUTE PSEUDO VISCOSITY");
    Real linear_coef = Options.ViscosityLinearCoef;
    Real quadratic_coef = Options.ViscosityQuadraticCoef;

    // Boucle sur les mailles du maillage
    // TODO: normalement on peut utiliser un 'Parallel.ForEach' mais
    // cela ne fonctionne pas avec '.NetCore' 2.2 avec un message
    // d'erreur indiquant que la méthode PInvoke ItemGroup.view() ne
    // fonctionne pas. A étudier mais en attendant on utilise
    // la version séquentielle.
#if false
    Parallel.ForEach(AllCells.SubViews(),
                     (ItemVectorView<Cell> sub_view) =>
        {
          foreach(Cell icell in sub_view){
            //Trace.Info("CELL={0}",icell.UniqueId);
            //const Integer cell_nb_node = cell.nbNode();
            Real rho = m_density[icell];

            // Calcul de la divergence de la vitesse
            Real delta_speed = 0.0;
            foreach(IndexedNode node in icell.Nodes){
              //for( int i=0; i<icell.NbNode; ++i )
              //delta_speed += Math.Dot(m_velocity[icell.Node(i)],m_cell_cqs[icell][i]);
              delta_speed += Math.Dot(m_velocity[node],m_cell_cqs[icell][node.Index]);
            }
            delta_speed /= m_volume[icell];

            // Capture uniquement les chocs
            bool shock = (Math.Min(0.0,delta_speed)<0.0);
            if (shock){
              Real sound_speed = m_sound_speed[icell];
              Real dx = m_caracteristic_length[icell];
              Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
              Real linear_viscosity = -rho*sound_speed* dx * delta_speed;
              Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
              m_cell_viscosity_force[icell] = scalar_viscosity;
            }
            else
              m_cell_viscosity_force[icell] = 0.0;
          }
        }
                     );
#else
    {
      foreach(Cell icell in AllCells()){
        //Trace.Info("CELL={0}",icell.UniqueId);
        //const Integer cell_nb_node = cell.nbNode();
        Real rho = m_density[icell];

        // Calcul de la divergence de la vitesse
        Real delta_speed = 0.0;
        foreach(IndexedNode node in icell.Nodes){
          //for( int i=0; i<icell.NbNode; ++i )
          //delta_speed += Math.Dot(m_velocity[icell.Node(i)],m_cell_cqs[icell][i]);
          delta_speed += Math.Dot(m_velocity[node],m_cell_cqs[icell][node.Index]);
        }
        delta_speed /= m_volume[icell];

        // Capture uniquement les chocs
        bool shock = (Math.Min(0.0,delta_speed)<0.0);
        if (shock){
          Real sound_speed = m_sound_speed[icell];
          Real dx = m_caracteristic_length[icell];
          Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
          Real linear_viscosity = -rho*sound_speed* dx * delta_speed;
          Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
          m_cell_viscosity_force[icell] = scalar_viscosity;
        }
        else
          m_cell_viscosity_force[icell] = 0.0;
      }
    }
#endif
  }

  public override void AddPseudoViscosityContribution()
  {
    //Trace.Info("ADD PSEUDO VISCOSITY");
    // Prise en compte des forces de viscosité si demandé
    bool add_viscosity_force = (Options.Viscosity!=TypesSimpleHydro.eViscosity.ViscosityNo);
    if (add_viscosity_force){
      foreach(Cell cell in AllCells()){
        Real scalar_viscosity = m_cell_viscosity_force[cell];
        foreach(IndexedNode node in cell.Nodes){
          //for( int i=0; i<cell.NbNode; ++i )
          m_force[node] += scalar_viscosity*m_cell_cqs[cell][node.Index];
        }
      }
    }
    m_force.Synchronize();
  }

  public override void ComputeVelocity()
  {
    //Trace.Info("COMPUTE VELOCITY");
    // Calcule l'impulsion aux noeuds
    foreach(Node node in AllNodes()){
      Real node_mass  = m_node_mass[node];

      Real3 old_velocity = m_velocity[node];
      Real3 new_velocity = old_velocity + (m_delta_t_n / node_mass) * m_force[node];
      
      m_velocity[node] = new_velocity;
    }
    m_velocity.Synchronize();
  }

  public override void ComputeViscosityWork()
  {
    // Calcul du travail des forces de viscosité dans une maille
    foreach(Cell cell in AllCells()){
      Real work = 0.0;
      Real scalar_viscosity = m_cell_viscosity_force[cell];
      foreach(IndexedNode node in cell.Nodes)
        work += Math.Dot(scalar_viscosity*m_cell_cqs[cell][node.Index],m_velocity[node]);
      m_cell_viscosity_work[cell] = work;
    }
  }
  
  public override void ApplyBoundaryCondition()
  {
    for( Integer i=0, nb=Options.BoundaryCondition.Length; i<nb; ++i){
      FaceGroup face_group = Options.BoundaryCondition[i].Surface;
      Real value = Options.BoundaryCondition[i].Value;
      TypesSimpleHydro.eBoundaryCondition type = Options.BoundaryCondition[i].Type;
      //Trace.Info("APPLY BOUNDARY CONDITION group={0}",face_group.name());
      // boucle sur les faces de la surface
      foreach(Face face in face_group){
        Integer nb_node = face.NbNode;
        // boucle sur les noeuds de la face
        for( Integer k=0; k<nb_node; ++k ){
          Node node = face.Node(k);
          Real3 v = m_velocity[node];
          switch(type) {
          case TypesSimpleHydro.eBoundaryCondition.VelocityX: v.x = value; break;
          case TypesSimpleHydro.eBoundaryCondition.VelocityY: v.y = value; break;
          case TypesSimpleHydro.eBoundaryCondition.VelocityZ: v.z = value; break;
          case TypesSimpleHydro.eBoundaryCondition.Unknown: break;
          }
          m_velocity[node] = v;
        }
      }
    }
  }

  public override void MoveNodes()
  {
    Real deltat_f = m_delta_t_f;

    foreach(Node node in AllNodes()){
      m_node_coord[node] += deltat_f * m_velocity[node];
    }
  }

  public override void UpdateDensity()
  {
    Real density_ratio_maximum = 0.0;

    foreach(Cell cell in AllCells()){
      Real old_density = m_density[cell];
      Real new_density = m_cell_mass[cell] / m_volume[cell];

      m_density[cell] = new_density;

      Real density_ratio = (new_density - old_density) / new_density;

      if (density_ratio_maximum<density_ratio)
        density_ratio_maximum = density_ratio;
    }
    
    m_density_ratio_maximum = density_ratio_maximum;
    m_density_ratio_maximum = ParallelMng().Reduce(Arcane.eReduceType.ReduceMax,m_density_ratio_maximum);

    m_density.Synchronize();
  }


  public override void ApplyEquationOfState()
  {
    Real deltatf = m_delta_t_f;
  
    bool add_viscosity_force = (Options.Viscosity!=TypesSimpleHydro.eViscosity.ViscosityNo);

    // Calcul de l'énergie interne
    foreach(Cell cell in AllCells()){
      Real adiabatic_cst = m_adiabatic_cst[cell];
      Real volume_ratio = m_volume[cell] / m_old_volume[cell];
      Real x = 0.5*(adiabatic_cst-1.0);
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      //Real denom2 = 1.0-(1.0/volume_ratio);
      /*info() << "RATIO " << ItemPrinter(*icell) << " n=" << numer_accrois_nrj
        << " d=" << denom_accrois_nrj << " volume_ratio=" << volume_ratio
        << " inv=" << (1.0/volume_ratio) << " x=" << x
        << " denom2=" << denom2 << " denom3=" << denom3 << " denom4=" << denom4;*/
      m_internal_energy[cell] *= numer_accrois_nrj/denom_accrois_nrj;
  
      // Prise en compte du travail des forces de viscosité 
      if (add_viscosity_force)
        m_internal_energy[cell] -= deltatf*m_cell_viscosity_work[cell] /  (m_cell_mass[cell]*denom_accrois_nrj);
    }

    // Synchronise l'énergie
    m_internal_energy.Synchronize();

    Options.EosModel.applyEOS(AllCells());
  }

  public override void ComputeDeltaT()
  {
    //m_global_deltat = 0.001;
    //m_global_time = 0.1;
    Real old_dt = m_global_deltat.Value;

    // Calcul du pas de temps pour le respect du critère de CFL
    
    Real minimum_aux = 1.0e100; //FloatInfo<Real>::maxValue();
    Real new_dt = 1.0e100; //FloatInfo<Real>::maxValue();

    //const bool new_test_cfl = true;

    foreach(Cell cell in AllCells()){
      Real cell_dx = m_caracteristic_length[cell];
      //Real density = m_density[icell];
      //Real pressure = m_pressure[icell];
      Real sound_speed = m_sound_speed[cell];
      Real dx_sound = cell_dx / sound_speed;
      minimum_aux = Math.Min(minimum_aux,dx_sound);
    }

    new_dt = Options.Cfl*minimum_aux;

    //Real cfl_dt = new_dt;

    // Pas de variations trop brutales à la hausse comme à la baisse
    Real max_dt = (1.0+Options.VariationSup)*old_dt;
    Real min_dt = (1.0-Options.VariationInf)*old_dt;

    new_dt = Math.Min(new_dt,max_dt);
    new_dt = Math.Max(new_dt,min_dt);

    //Real variation_min_max_dt = new_dt;

    // control de l'accroissement relatif de la densité
    Real dgr = Options.DensityGlobalRatio;
    if (m_density_ratio_maximum>dgr)
      new_dt = Math.Min(old_dt*dgr/m_density_ratio_maximum,new_dt);

    new_dt = ParallelMng().Reduce(eReduceType.ReduceMin,new_dt);

    // respect des valeurs min et max imposées par le fichier de données .plt
    new_dt = Math.Min(new_dt,Options.DeltatMax);
    new_dt = Math.Max(new_dt,Options.DeltatMin);

    //Real data_min_max_dt = new_dt;
    
    // Le dernier calcul se fait exactement au temps stopTime()
    {
      Real stop_time  = Options.FinalTime;
      bool not_yet_finish = ( m_global_time.Value < stop_time);
      bool too_much = ( (m_global_time.Value+new_dt) > stop_time);
      
      if ( not_yet_finish && too_much ){
        new_dt = stop_time - m_global_time.Value;
        SubDomain().TimeLoopMng().StopComputeLoop(true);
      }
    }
    
    // Mise à jour des variables
    m_old_dt_f = old_dt;
    m_delta_t_n = 0.5*(old_dt+new_dt);
    m_delta_t_f = new_dt;
    m_global_deltat.Value = new_dt;
    //Trace.Info("END COMPUTE DELTAT @!!!!!!!!!!!!!");
  }
  
  public void computeCQs(Real3ConstArrayView node_coord,Real3ConstArrayView face_coord,Cell cell)
  {
    Real3 c0 = face_coord[0];
    Real3 c1 = face_coord[1];
    Real3 c2 = face_coord[2];
    Real3 c3 = face_coord[3];
    Real3 c4 = face_coord[4];
    Real3 c5 = face_coord[5];

    Real demi = 0.5;
    Real five = 5.0;

    // Calcul des normales face 1 :
    Real3 n1a04 = demi * Math.VecMul(node_coord[0] - c0 , node_coord[3] - c0);
    Real3 n1a03 = demi * Math.VecMul(node_coord[3] - c0 , node_coord[2] - c0);
    Real3 n1a02 = demi * Math.VecMul(node_coord[2] - c0 , node_coord[1] - c0);
    Real3 n1a01 = demi * Math.VecMul(node_coord[1] - c0 , node_coord[0] - c0);

    // Calcul des normales face 2 :
    Real3 n2a05 = demi * Math.VecMul(node_coord[0] - c1 , node_coord[4] - c1);
    Real3 n2a12 = demi * Math.VecMul(node_coord[4] - c1 , node_coord[7] - c1);
    Real3 n2a08 = demi * Math.VecMul(node_coord[7] - c1 , node_coord[3] - c1);
    Real3 n2a04 = demi * Math.VecMul(node_coord[3] - c1 , node_coord[0] - c1);

    // Calcul des normales face 3 :
    Real3 n3a01 = demi * Math.VecMul(node_coord[0] - c2 , node_coord[1] - c2);
    Real3 n3a06 = demi * Math.VecMul(node_coord[1] - c2 , node_coord[5] - c2);
    Real3 n3a09 = demi * Math.VecMul(node_coord[5] - c2 , node_coord[4] - c2);
    Real3 n3a05 = demi * Math.VecMul(node_coord[4] - c2 , node_coord[0] - c2);

    // Calcul des normales face 4 :
    Real3 n4a09 = demi * Math.VecMul(node_coord[4] - c3 , node_coord[5] - c3);
    Real3 n4a10 = demi * Math.VecMul(node_coord[5] - c3 , node_coord[6] - c3);
    Real3 n4a11 = demi * Math.VecMul(node_coord[6] - c3 , node_coord[7] - c3);
    Real3 n4a12 = demi * Math.VecMul(node_coord[7] - c3 , node_coord[4] - c3);
	
    // Calcul des normales face 5 :
    Real3 n5a02 = demi * Math.VecMul(node_coord[1] - c4 , node_coord[2] - c4);
    Real3 n5a07 = demi * Math.VecMul(node_coord[2] - c4 , node_coord[6] - c4);
    Real3 n5a10 = demi * Math.VecMul(node_coord[6] - c4 , node_coord[5] - c4);
    Real3 n5a06 = demi * Math.VecMul(node_coord[5] - c4 , node_coord[1] - c4);
      
    // Calcul des normales face 6 :
    Real3 n6a03 = demi * Math.VecMul(node_coord[2] - c5 , node_coord[3] - c5);
    Real3 n6a08 = demi * Math.VecMul(node_coord[3] - c5 , node_coord[7] - c5);
    Real3 n6a11 = demi * Math.VecMul(node_coord[7] - c5 , node_coord[6] - c5);
    Real3 n6a07 = demi * Math.VecMul(node_coord[6] - c5 , node_coord[2] - c5);

    Real real_1div12 = 1.0 / 12.0;

    // Calcul des résultantes aux sommets :
    Real3ArrayView v = m_cell_cqs[cell];
    v[0] = (five*(n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) +
            (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09))*real_1div12;
    v[1] = (five*(n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) +
            (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07))*real_1div12;
    v[2] = (five*(n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) +
            (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08))*real_1div12;
    v[3] = (five*(n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) +
            (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11))*real_1div12;
    v[4] = (five*(n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) +
            (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11))*real_1div12;
    v[5] = (five*(n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) +
            (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02))*real_1div12;
    v[6] = (five*(n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) +
            (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08))*real_1div12;
    v[7] = (five*(n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) +
            (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03))*real_1div12;
  }

  public override void ComputeGeometricValues()
  {
    //Trace.Info("COMPUTE GEOMETRIC VALUES BEGIN");
    m_old_volume.Copy(m_volume);

    Real3Array tmp_coord = new Real3Array(8+6);
    Real3ArrayView tmp_coord_view = tmp_coord.View;
    // Copie locale des coordonnées des sommets d'une maille
    Real3ArrayView coord = tmp_coord_view.SubView(0,8);
    // Coordonnées des centres des faces
    Real3ArrayView face_coord = tmp_coord_view.SubView(8,6);

    foreach(Cell cell in AllCells()){

      // Recopie les coordonnées en local (pour le cache)
      foreach(IndexedNode node in cell.Nodes){
        //Trace.Info("CELL UID={0} I={1} NODE={2} COORD={3}",
        //                  cell.UniqueId,node.Index,((Node)node).UniqueId,m_node_coord[node].x);
        coord[node.Index] = m_node_coord[node];
      }

      // Calcul les coordonnées des centres des faces
      face_coord[0] = 0.25 * ( coord[0] + coord[3] + coord[2] + coord[1] );
      face_coord[1] = 0.25 * ( coord[0] + coord[4] + coord[7] + coord[3] );
      face_coord[2] = 0.25 * ( coord[0] + coord[1] + coord[5] + coord[4] );
      face_coord[3] = 0.25 * ( coord[4] + coord[5] + coord[6] + coord[7] );
      face_coord[4] = 0.25 * ( coord[1] + coord[2] + coord[6] + coord[5] );
      face_coord[5] = 0.25 * ( coord[2] + coord[3] + coord[7] + coord[6] );

      // Calcule la longueur caractéristique de la maille.
      {
        Real3 median1 = face_coord[0]-face_coord[3];
        Real3 median2 = face_coord[2]-face_coord[5];
        Real3 median3 = face_coord[1]-face_coord[4];
        Real d1 = median1.Abs();
        Real d2 = median2.Abs();
        Real d3 = median3.Abs();

        Real dx_numerator   = d1*d2*d3;
        Real dx_denominator = d1*d2 + d1*d3 + d2*d3;
        m_caracteristic_length[cell] = dx_numerator / dx_denominator;
      }

      // Calcule les résultantes aux sommets
      computeCQs(coord.ConstView,face_coord.ConstView,cell);

      // Calcule le volume de la maille
      {
        Real volume = 0.0;
        for( Integer i_node=0; i_node<8; ++i_node )
          volume += Math.Dot(coord[i_node],m_cell_cqs[cell][i_node]);
        volume /= 3.0;
        m_volume[cell] = volume;
      }
    }
    tmp_coord.Dispose();
    //Trace.Info("COMPUTE GEOMETRIC VALUES END");
  }

  void _PrintInfos()
  {
    foreach(int i in Options.TestInt32){
      Trace.Info("VALUE = {0}",i);
    }
    foreach(ItemGroup group in Options.Volume){
      Trace.Info("GROUP name={0} type={1}",group.Name(),group.ItemKind());
    }

    foreach(Face face in AllFaces()){
      Console.Write("BOUNDARY FACE uid={0} nb_cell={1}",face.UniqueId,face.NbCell);
      Cell back_cell = face.BackCell;
      Cell front_cell = face.FrontCell;
      Trace.Info("BACK CELL IS NULL? {0}",back_cell.IsNull);
      if (!back_cell.IsNull)
        Trace.Info(" BACK CELL uid={0}",back_cell.UniqueId);
      Trace.Info("FRONT CELL IS NULL? {0}",front_cell.IsNull);
      if (!front_cell.IsNull)
        Trace.Info(" FRONT CELL uid={0}",front_cell.UniqueId);
      Trace.Info("");
    }
  }
}
