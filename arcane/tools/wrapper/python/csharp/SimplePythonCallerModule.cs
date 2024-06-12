using Arcane;
using System;
using Python.Runtime;
using System.Threading.Tasks;
using Numpy;

[Arcane.Module("SimplePythonCallerModule","0.0.1")]
class SimplePythonCallerModule
: Arcane.BasicModule
{
  public SimplePythonCallerModule(ModuleBuildInfo infos) : base(infos)
  {
    Console.WriteLine("Create SimplePythonCallerModule");
    _AddEntryPoint("Main1",MyEntryPoint,IEntryPoint.WComputeLoop,0);
    _AddEntryPoint("OnExit",OnExit,IEntryPoint.WExit,0);
    PythonEngine.Initialize();
  }

  void MyEntryPoint()
  {
    Console.WriteLine("Test calling Python");
    
    int c = -1;
    using (Py.GIL())
    using (var scope = Py.CreateScope())
    {
      var p1 = np.array<double>(new double[]{1, 3.423, 1.5, 12.5});
      scope.Set("my_array",p1);
      //scope.Set(parameterName, parameter.ToPython());
      scope.Exec("a=my_array.size\nb=5\nc=a+b\nprint(c)\n");
      c = scope.Get<int>("c");
    }
    Console.WriteLine("C={0}",c);
    Console.WriteLine("End setup python");
  }

  void OnExit()
  {
    Console.WriteLine("Exiting python");
    PythonEngine.Shutdown();
  }
}
