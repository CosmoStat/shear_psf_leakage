# +                                                                             
# Plot PSF auto-correlation terms                                               
                                                                                
Xi_11 = []                                                                      
Xi_22 = []                                                                      
for ndx in range(len(theta)):                                                   
    Xi_11.append(obj.Xi_pp_ufloat[ndx][0, 0])                                   
    Xi_22.append(obj.Xi_pp_ufloat[ndx][1, 1])                                   
                                                                                
t_1 = obj.get_alpha_ufloat(0, 0) ** 2*  Xi_11
