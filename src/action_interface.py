import pyautogui


class Actor():

  encodings=['10000000','01000000','00100000','00010000','00001000','00000100','00000010','00000001']

  def get_encs(self,prediction):
    active_encoding = str(prediction)
    return active_encoding

  def act(self,active_encoding):

        if(active_encoding==self.encodings[0]):
            pyautogui.moveRel(-40, 0, duration = 0.001)

        elif(active_encoding==self.encodings[1]):
            pyautogui.moveRel(40, 0, duration = 0.001)

        elif(active_encoding==self.encodings[2]):
            pyautogui.moveRel(0, -40, duration = 0.001)

        elif(active_encoding==self.encodings[3]):
            pyautogui.moveRel(0, 40, duration = 0.001)

        elif(active_encoding==self.encodings[4]):
            pyautogui.moveRel(-20, 0, duration = 0.001)

        elif(active_encoding==self.encodings[5]):
            pyautogui.moveRel(20, 0, duration = 0.001)

        elif(active_encoding==self.encodings[6]):
            pyautogui.moveRel(0,-20, duration = 0.001)

        elif(active_encoding==self.encodings[7]):
            pyautogui.moveRel(0, 20, duration = 0.001)

        elif(type(active_encoding)==None):
            a=1

        else:
            print('Error')
        
  def test_action(self,prediction):
      active_encoding = self.get_encs(prediction)
      self.act(active_encoding)
      return