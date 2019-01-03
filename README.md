# Power-line-fault-detection
Medium voltage overhead power lines run for hundreds of miles to supply power to cities. These great distances make it expensive to manually inspect the lines for damage that doesn't immediately lead to a power outage, such as a tree branch hitting the line or a flaw in the insulator. These modes of damage lead to a phenomenon known as partial discharge — an electrical discharge which does not bridge the electrodes between an insulation system completely. Partial discharges slowly damage the power line, so left unrepaired they will eventually lead to a power outage or start a fire.
I am trying to do signal processing to find the pattern of occurence of the partial discharge.
<html>
  <h1> Points I am trying to remember</h1>
  <body>
    <h3>Try to do FFT and multiply with conjugate of two signals.</h3>
    <h3>You could compute the covariance that measures how much a signal is similar to another. For example in Matlab, you can do that with the function cov().</h3>
