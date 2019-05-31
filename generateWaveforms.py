import pylab
from pycbc.waveform import get_td_waveform

for mass_1 in range(1, 300):
    for mass_2 in range(1, 300):
        hp, hc = get_td_waveform(approximant='IMRPhenomC',
                                     mass1=mass_1/100,
                                     mass2=mass_2/100,
                                     spin1z=0,
                                     delta_t=1.0/4096,
                                     f_lower=10)

        f = open('NS Waveforms/' + str(mass_1) + '_' + str(mass_2) + '.dat', 'w+')

        for j in range(len(hp.sample_times)):
            f.write(str(hp.sample_times[j]) + ' ' + str(hp[j]) + '\n')

        f.close()

#         pylab.plot(hp.sample_times, hp, label=str(mass_1))
#0
#
# pylab.ylabel('Strain')
# pylab.xlabel('Time (s)')
# pylab.legend()
# pylab.show()
