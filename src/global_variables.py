# path to label dir
DRIVE_PATH = "drive/MyDrive/Action Spotting CSE 610/labels/"
LOCAL_PATH = "labels/"

# Event name to label index for SoccerNet-V3
EVENT_DICTIONARY_V2 = {
	"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,
	"Offside":4,"Shots on target":5,"Shots off target":6,
	"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
    "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,
    "Yellow card":14,"Red card":15,"Yellow->red card":16
}

INVERSE_EVENT_DICTIONARY_V2 = {0:"Penalty",1:"Kick-off",2:"Goal",3:"Substitution",4:"Offside",5:"Shots on target",
                                6:"Shots off target",7:"Clearance",8:"Ball out of play",9:"Throw-in",10:"Foul",
                                11:"Indirect free-kick",12:"Direct free-kick",13:"Corner",14:"Yellow card"
                                ,15:"Red card",16:"Yellow->red card"}

MOD_EVENT_DICTIONARY_V2 = {
	"Penalty":0,
	"Kick-off":1,
	"Goal":2,"Shots on target":2,"Shots off target":2,"Corner":2,
	"Substitution":3,
	"Yellow card":4,"Red card":4,"Yellow->red card":4,"Offside":4,
	"Clearance":5,"Ball out of play":5,"Throw-in":5,
	"Foul":6,"Indirect free-kick":6,"Direct free-kick":6,
}
