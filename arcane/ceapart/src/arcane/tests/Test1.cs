using System;
using Arcane;
using Arcane.Materials;

[Arcane.Service("DotNetMaterialTest1",typeof(Arcane.IUnitTest))]
public class DotNetMaterialTest1 : Arcane.IUnitTest_WrapperService
{
  IMeshMaterialMng_Ref m_mesh_material_mng;
  MaterialVariableCellReal m_mat_density;

  public DotNetMaterialTest1(ServiceBuildInfo bi) : base(bi)
  {
  }

  public override void InitializeTest()
  {
    m_mesh_material_mng = IMeshMaterialMng.GetTrueReference(Mesh().Handle());
    m_mat_density = new MaterialVariableCellReal(new VariableBuildInfo(Mesh(),"Density"));
    Console.WriteLine("[C#] Initialize test");
  }

  public override void ExecuteTest()
  {
    var mm = m_mesh_material_mng.Get();
    Console.WriteLine("[C#] EXECUTE TEST1 mesh_material_mng name={0}",mm.Name());

    var envs = mm.Environments();
    for( int i=0; i<envs.Size; ++i ){
      IMeshEnvironment env = envs[i];
      Console.WriteLine("ENV i={0} name={1}",i,env.Name());
    }

    foreach(IMeshEnvironment env in mm.Environments()){
      Console.WriteLine("ENV name={0}",env.Name());
    }
    foreach(IMeshMaterial mat in mm.Materials()){
      Console.WriteLine("MAT name={0}",mat.Name());
    }
    foreach(IMeshComponent cmp in mm.Components()){
      Console.WriteLine("CMP name={0}",cmp.Name());
    }
    foreach(IMeshComponent env in mm.EnvironmentsAsComponents()){
      Console.WriteLine("ENV_AS_CMP name={0}",env.Name());
    }
    foreach(IMeshComponent mat in mm.MaterialsAsComponents()){
      Console.WriteLine("MAT_AS_CMP name={0}",mat.Name());
    }

    IMeshEnvironment env1 = mm.FindEnvironment("ENV1");
    var env1_view = env1.EnvView();
    Console.WriteLine("NB_ITEM = '{0}'",env1_view.NbItem());

    foreach(ComponentItem ci in env1){
      Console.WriteLine("[C#] Component! {0} density={1}",ci.GlobalCell.UniqueId,m_mat_density[ci]);
    }
    foreach(EnvItem ci in env1){
      Console.WriteLine("[C#] Env! {0} density={1}",ci.GlobalCell.UniqueId,m_mat_density[ci]);
    }
    foreach(IMeshComponent cmp in mm.EnvironmentsAsComponents()){
      Console.WriteLine("CMP name={0} nb_pure={1} nb_impure={2}",cmp.Name(),cmp.PureItems().NbItem(),cmp.ImpureItems().NbItem());
      foreach(ComponentItem ci in cmp){
        Console.WriteLine("[C#] Component! {0} density={1}",ci.GlobalCell.UniqueId,m_mat_density[ci]);
      }
    }
    foreach(IMeshMaterial mat in mm.Materials()){
      Console.WriteLine("XMAT name={0} nb_pure={1} nb_impure={2}",mat.Name(),mat.PureMatItems().NbItem(),mat.ImpureMatItems().NbItem());
      foreach(MatItem mc in mat){
        Console.WriteLine("[C#] Mat! {0} density={1}",mc.GlobalCell.UniqueId,m_mat_density[mc]);
      }
    }
  }

  public override void FinalizeTest()
  {
  }
}
