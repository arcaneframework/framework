<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test MiniWeather using original serial version</title>
  <timeloop>MiniWeatherLoop</timeloop>
 </arcane>

 <mesh>
  <meshgenerator><sod><x>50</x><y>5</y><z>5</z></sod></meshgenerator>
 </mesh>

 <arcane-checkpoint>
  <do-dump-at-end>false</do-dump-at-end>
 </arcane-checkpoint>

 <mini-weather>
  <nb-cell-x>400</nb-cell-x>
  <nb-cell-z>200</nb-cell-z>
  <final-time>2.0</final-time>
  <implementation name="MiniWeatherArray" />
 </mini-weather>
</case>
